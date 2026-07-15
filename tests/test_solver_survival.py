import unittest

import numpy as np

from block_blast_solver.modules import solver
from block_blast_solver.modules.visualizer import summarize_move_sequence


def empty_board():
    return np.zeros((8, 8), dtype=np.uint8)


def piece(matrix):
    return np.array(matrix, dtype=np.uint8)


class SurvivalSolverTests(unittest.TestCase):
    def test_future_fit_score_rewards_open_board(self):
        constrained_board = np.ones((8, 8), dtype=np.uint8)
        constrained_board[0, 0] = 0
        constrained_board[7, 7] = 0

        open_score = solver.count_future_piece_fits_jit(0)
        constrained_mask = 0
        for r in range(8):
            for c in range(8):
                if constrained_board[r, c] == 1:
                    constrained_mask |= 1 << (r * 8 + c)
        constrained_score = solver.count_future_piece_fits_jit(constrained_mask)

        self.assertGreater(open_score, constrained_score)

    def test_empty_region_scoring_prefers_connected_space(self):
        connected_board = np.ones((8, 8), dtype=np.uint8)
        connected_board[0:4, 0:4] = 0

        fragmented_board = np.ones((8, 8), dtype=np.uint8)
        for r in range(4):
            for c in range(4):
                fragmented_board[r * 2, c * 2] = 0

        connected_mask = 0
        fragmented_mask = 0
        for r in range(8):
            for c in range(8):
                if connected_board[r, c] == 1:
                    connected_mask |= 1 << (r * 8 + c)
                if fragmented_board[r, c] == 1:
                    fragmented_mask |= 1 << (r * 8 + c)

        connected_largest, connected_small, connected_count = solver.calculate_empty_regions_jit(np.int64(np.uint64(connected_mask)))
        fragmented_largest, fragmented_small, fragmented_count = solver.calculate_empty_regions_jit(np.int64(np.uint64(fragmented_mask)))

        self.assertGreater(connected_largest, fragmented_largest)
        self.assertLess(connected_small, fragmented_small)
        self.assertLess(connected_count, fragmented_count)

    def test_never_places_on_occupied_cells(self):
        board = empty_board()
        board[0, 0] = 1
        board[0, 1] = 1
        board[1, 0] = 1

        pieces = [piece([[1, 1], [1, 1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNotNone(moves)
        move = moves[0]
        row = move["row"]
        col = move["col"]
        placed = pieces[move["slot_index"]]

        for r in range(placed.shape[0]):
            for c in range(placed.shape[1]):
                if placed[r, c] == 1:
                    self.assertEqual(board[row + r, col + c], 0)
        self.assertGreater(score, -1e8)

    def test_preserves_open_three_by_three_space(self):
        board = np.ones((8, 8), dtype=np.uint8)
        board[0:3, 0:3] = 0
        board[5:8, 5:8] = 0
        board[6, 6] = 1

        pieces = [piece([[1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNotNone(moves)
        move = moves[0]
        self.assertFalse(0 <= move["row"] <= 2 and 0 <= move["col"] <= 2)
        self.assertGreater(score, -1e8)

    def test_prefers_clear_when_board_is_constrained(self):
        board = empty_board()
        board[0, 0:7] = 1
        board[1:8, :] = 1
        board[1, 0] = 0
        board[2, 0] = 0

        pieces = [piece([[1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNotNone(moves)
        self.assertEqual((moves[0]["row"], moves[0]["col"]), (0, 7))
        self.assertGreater(score, -1e8)

    def test_prefers_multi_step_clear_over_immediate_single_line_clear(self):
        board = empty_board()
        board[7, 0:3] = 1

        pieces = [
            piece([[1, 1, 1, 1, 1]]),
            piece([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            piece([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ]

        moves, score = solver.solve(board, pieces)

        self.assertIsNotNone(moves)
        outcomes, total_clears, final_filled = summarize_move_sequence(board, pieces, moves)
        self.assertEqual({move["slot_index"] for move in moves}, {0, 1, 2})
        self.assertEqual([outcome["clear_count"] for outcome in outcomes], [0, 0, 3])
        self.assertEqual(total_clears, 3)
        self.assertEqual(final_filled, 2)
        self.assertGreater(score, -1e8)

    def test_returns_none_when_no_legal_move_exists(self):
        board = np.ones((8, 8), dtype=np.uint8)
        pieces = [piece([[1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNone(moves)
        self.assertLess(score, -1e8)

    def test_rejects_invalid_board_and_piece_inputs(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            solver.solve(np.zeros((7, 8), dtype=np.uint8), [piece([[1]])])
        with self.assertRaisesRegex(ValueError, "binary"):
            solver.solve(np.full((8, 8), 2, dtype=np.uint8), [piece([[1]])])
        with self.assertRaisesRegex(ValueError, "non-empty binary"):
            solver.solve(empty_board(), [np.zeros((1, 1), dtype=np.uint8)])
        with self.assertRaisesRegex(ValueError, "at most three"):
            solver.solve(empty_board(), [piece([[1]])] * 4)


if __name__ == "__main__":
    unittest.main()
