import unittest

import numpy as np

from block_blast_solver.modules import solver


class StreakCountingTests(unittest.TestCase):
    def test_simulate_next_set_returns_clearing_placement_count(self):
        catalog_masks, catalog_shapes, _ = solver.get_monte_carlo_piece_catalog()
        samples = solver.get_next_piece_samples(1)
        survived, _, _, placed, clearing = solver.simulate_next_set_survival_jit(
            np.int64(0), samples[0], catalog_masks, catalog_shapes
        )
        self.assertTrue(survived)
        self.assertEqual(placed, 3)
        self.assertGreaterEqual(clearing, 0)
        self.assertLessEqual(clearing, placed)


class MonteCarloStreakContinuityTests(unittest.TestCase):
    def test_evaluator_returns_streak_continuity_pct(self):
        result = solver.evaluate_monte_carlo_survival_jit(np.int64(0), 2)
        self.assertEqual(len(result), 5)
        score, survival_pct, clear_routes, future_fits, streak_continuity = result
        self.assertGreaterEqual(streak_continuity, 0.0)
        self.assertLessEqual(streak_continuity, 1.0)

    def test_dead_board_streak_continuity_is_zero(self):
        full_board = np.int64(-1)
        _, _, _, _, streak_continuity = solver.evaluate_monte_carlo_survival_jit(full_board, 2)
        self.assertEqual(streak_continuity, 0.0)


def _empty_board():
    return np.zeros((8, 8), dtype=np.int32)


def _piece_1x1():
    return np.array([[1]], dtype=np.int32)


class StreakScoringInRecursionTests(unittest.TestCase):
    def test_diagnostics_dict_contains_next_streak_pct(self):
        # Constrained board to keep search fast.
        board = np.ones((8, 8), dtype=np.int32)
        board[7, 0:7] = 0
        board[0, 7] = 0
        pieces = [_piece_1x1(), _piece_1x1(), _piece_1x1()]
        _, _, diagnostics = solver.solve_with_diagnostics(board, pieces)
        self.assertIn("next_streak_pct", diagnostics)
        self.assertGreaterEqual(diagnostics["next_streak_pct"], 0.0)
        self.assertLessEqual(diagnostics["next_streak_pct"], 1.0)


class StreakVsFatClearTests(unittest.TestCase):
    def test_break_penalty_dominates_single_fat_clear(self):
        # Plan A: three 1x1 pieces each clear one row (perfect streak +1100).
        # Plan B: three pieces in row 7 (no clears, no streak).
        # Solver must pick Plan A for the streak.
        board = np.zeros((8, 8), dtype=np.int32)
        # Plan A: rows 1, 3, 5 are 7-filled (only col 0 empty -> clear row).
        for r in (1, 3, 5):
            board[r, 1:8] = 1
        # Plan B "neutral" cells: row 7 has cells 0..3 filled, 4..7 empty.
        # Placing 1x1 at (7,4), (7,5), (7,6) leaves (7,7) empty -> no clear.
        board[7, 0:4] = 1

        pieces = [_piece_1x1(), _piece_1x1(), _piece_1x1()]
        moves, score, diagnostics = solver.solve_with_diagnostics(board, pieces)

        self.assertIsNotNone(moves)
        from block_blast_solver.modules.visualizer import summarize_move_sequence
        outcomes, total_clears, _ = summarize_move_sequence(board, pieces, moves)
        per_placement_clears = [o["clear_count"] for o in outcomes]
        self.assertTrue(
            all(c >= 1 for c in per_placement_clears),
            f"Expected every placement to clear (perfect streak), got per-placement clears={per_placement_clears} moves={moves} score={score}",
        )


if __name__ == "__main__":
    unittest.main()
