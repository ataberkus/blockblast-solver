import unittest

import cv2
import numpy as np

from block_blast_solver import config
from block_blast_solver.modules import vision

BOARD_ROI = [0.1, 0.1, 0.8, 0.8]


def draw_board(filled_cells=(), empty_value=45, occupied_value=185):
    frame = np.full((400, 400, 3), 20, dtype=np.uint8)
    x, y, width = 40, 40, 320
    cell_size = width // 8
    for row in range(8):
        for col in range(8):
            value = occupied_value if (row, col) in filled_cells else empty_value
            x1 = x + col * cell_size
            y1 = y + row * cell_size
            cv2.rectangle(frame, (x1, y1), (x1 + cell_size - 1, y1 + cell_size - 1), (value, value, value), -1)
    return frame


class VisionBoardTests(unittest.TestCase):
    def setUp(self):
        self.previous_roi = config.BOARD_ROI
        config.BOARD_ROI = BOARD_ROI

    def tearDown(self):
        config.BOARD_ROI = self.previous_roi

    def test_uniform_dark_board_is_valid_empty_state(self):
        board, occluded = vision.get_board_state(draw_board())

        self.assertFalse(occluded)
        self.assertEqual(int(np.sum(board)), 0)

    def test_detects_sparse_occupied_cells(self):
        expected_cells = {(0, 0), (3, 4), (7, 7)}
        board, occluded = vision.get_board_state(draw_board(expected_cells))

        self.assertFalse(occluded)
        self.assertEqual({tuple(cell) for cell in np.argwhere(board == 1)}, expected_cells)

    def test_detects_single_empty_cell_on_nearly_full_board(self):
        empty_cell = (6, 2)
        filled_cells = {(row, col) for row in range(8) for col in range(8)} - {empty_cell}
        board, occluded = vision.get_board_state(draw_board(filled_cells))

        self.assertFalse(occluded)
        self.assertEqual(int(np.sum(board)), 63)
        self.assertEqual(int(board[empty_cell]), 0)

    def test_uniform_bright_crop_is_occluded(self):
        board, occluded = vision.get_board_state(draw_board(empty_value=210))

        self.assertTrue(occluded)
        self.assertEqual(int(np.sum(board)), 0)

    def test_invalid_roi_fails_closed(self):
        config.BOARD_ROI = [0.9, 0.9, 0.5, 0.5]

        board, occluded = vision.get_board_state(draw_board())

        self.assertTrue(occluded)
        self.assertEqual(int(np.sum(board)), 0)


if __name__ == "__main__":
    unittest.main()
