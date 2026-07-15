import unittest

import numpy as np

from block_blast_solver.state import AppState, state_changed


class AppStateTests(unittest.TestCase):
    def test_requires_consecutive_matching_frames(self):
        state = AppState()
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [np.ones((1, 1), dtype=np.uint8), None, None]

        self.assertIsNone(state.observe(board, pieces, 2))
        update = state.observe(board, pieces, 2)

        self.assertIsNotNone(update)
        self.assertTrue(update.changed)

    def test_changed_detection_resets_stability(self):
        state = AppState()
        board = np.zeros((8, 8), dtype=np.uint8)
        changed_board = board.copy()
        changed_board[0, 0] = 1

        state.observe(board, [None, None, None], 2)
        self.assertIsNone(state.observe(changed_board, [None, None, None], 2))
        self.assertEqual(state.stable_frame_count, 1)

    def test_occlusion_resets_stability_but_preserves_advice(self):
        state = AppState()
        board = np.zeros((8, 8), dtype=np.uint8)
        state.last_moves = [{"slot_index": 0, "row": 1, "col": 2}]
        state.observe(board, [None, None, None], 2)

        self.assertIsNone(state.observe(board, [None, None, None], 2, occluded=True))

        self.assertEqual(state.stable_frame_count, 0)
        self.assertIsNotNone(state.last_moves)

    def test_accept_and_reset_copy_mutable_inputs(self):
        state = AppState()
        board = np.zeros((8, 8), dtype=np.uint8)
        piece = np.ones((1, 1), dtype=np.uint8)
        state.accept(board, [piece, None, None], [], 12.0, {"risk_level": "low"})
        board[0, 0] = 1
        piece[0, 0] = 0

        self.assertEqual(int(state.last_board[0, 0]), 0)
        self.assertEqual(int(state.last_pieces[0][0, 0]), 1)

        state.reset()
        self.assertIsNone(state.last_moves)
        self.assertEqual(state.stable_frame_count, 0)

    def test_state_changed_compares_piece_content(self):
        board = np.zeros((8, 8), dtype=np.uint8)
        self.assertFalse(state_changed(board, board.copy(), [None], [None]))
        self.assertTrue(
            state_changed(
                board,
                board.copy(),
                [np.array([[1]], dtype=np.uint8)],
                [np.array([[1, 1]], dtype=np.uint8)],
            )
        )


if __name__ == "__main__":
    unittest.main()
