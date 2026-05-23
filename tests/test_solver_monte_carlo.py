import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER_ROOT = os.path.join(ROOT, "block_blast_solver")
if SOLVER_ROOT not in sys.path:
    sys.path.insert(0, SOLVER_ROOT)

from modules import solver


def empty_board():
    return np.zeros((8, 8), dtype=np.uint8)


def piece(matrix):
    return np.array(matrix, dtype=np.uint8)


def board_mask(board):
    mask = 0
    for row in range(8):
        for col in range(8):
            if board[row, col] == 1:
                mask |= 1 << (row * 8 + col)
    return mask


class MonteCarloSolverTests(unittest.TestCase):
    def test_piece_catalog_contains_rotated_and_mirrored_families(self):
        masks, shapes, families = solver.get_monte_carlo_piece_catalog()

        self.assertGreaterEqual(masks.shape[0], 24)
        self.assertEqual(masks.shape[0], shapes.shape[0])
        self.assertEqual(masks.shape[0], families.shape[0])
        self.assertIn(9, set(int(family) for family in families))
        self.assertIn(25, set(int(family) for family in families))

    def test_next_piece_samples_are_deterministic(self):
        first = solver.get_next_piece_samples(24)
        second = solver.get_next_piece_samples(24)

        self.assertEqual(first.shape, (24, 3))
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(int(np.max(first)), 0)

    def test_monte_carlo_rewards_more_playable_next_sets(self):
        open_board = empty_board()
        trapped_board = np.ones((8, 8), dtype=np.uint8)
        trapped_board[0, 0] = 0
        trapped_board[0, 1] = 0
        trapped_board[1, 0] = 0
        trapped_board[7, 7] = 0

        open_score, open_survival, open_routes, open_fits, _ = solver.evaluate_monte_carlo_survival_jit(board_mask(open_board), 24)
        trapped_score, trapped_survival, trapped_routes, trapped_fits, _ = solver.evaluate_monte_carlo_survival_jit(board_mask(trapped_board), 24)

        self.assertGreater(open_survival, trapped_survival)
        self.assertGreater(open_fits, trapped_fits)
        self.assertGreater(open_score, trapped_score)
        self.assertGreaterEqual(open_routes, trapped_routes)

    def test_solver_prefers_high_survival_board_over_cosmetic_immediate_clear(self):
        board = empty_board()
        board[7, 0:7] = 1
        board[0:5, 0:5] = 1
        board[0:3, 5:8] = 0
        board[5:8, 5:8] = 0

        pieces = [
            piece([[1]]),
            piece([[1, 1], [1, 0]]),
            piece([[1, 1], [1, 0]]),
        ]

        moves, score, diagnostics = solver.solve_with_diagnostics(board, pieces)

        self.assertIsNotNone(moves)
        self.assertGreater(score, -1e8)
        self.assertGreaterEqual(diagnostics["next_survival_pct"], 20.0)
        self.assertGreaterEqual(diagnostics["future_fits"], 1)

    def test_diagnostics_are_zero_when_no_legal_move_exists(self):
        board = np.ones((8, 8), dtype=np.uint8)
        pieces = [piece([[1]]), None, None]

        moves, score, diagnostics = solver.solve_with_diagnostics(board, pieces)

        self.assertIsNone(moves)
        self.assertLess(score, -1e8)
        self.assertEqual(diagnostics["risk_level"], "dead")
        self.assertEqual(diagnostics["next_survival_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
