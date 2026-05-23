import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER_ROOT = os.path.join(ROOT, "block_blast_solver")
if SOLVER_ROOT not in sys.path:
    sys.path.insert(0, SOLVER_ROOT)

import config
from modules import vision


def draw_inventory(shapes, block_size=23):
    frame_h = 320
    frame_w = 480
    roi_x = 60
    roi_y = 170
    roi_w = 360
    roi_h = 130
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[:] = (55, 135, 195)
    config.PIECES_ROI = [roi_x / frame_w, roi_y / frame_h, roi_w / frame_w, roi_h / frame_h]

    colors = [(220, 160, 30), (30, 120, 220), (210, 40, 160)]
    slot_w = roi_w // 3
    for slot, shape in enumerate(shapes):
        shape_w = shape.shape[1] * block_size
        shape_h = shape.shape[0] * block_size
        start_x = roi_x + slot * slot_w + (slot_w - shape_w) // 2
        start_y = roi_y + (roi_h - shape_h) // 2
        for row in range(shape.shape[0]):
            for col in range(shape.shape[1]):
                if shape[row, col] == 0:
                    continue

                x1 = start_x + col * block_size
                y1 = start_y + row * block_size
                x2 = x1 + block_size - 1
                y2 = y1 + block_size - 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[slot], -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (250, 250, 250), 2)
                cv2.line(frame, (x1, y2 - 4), (x2, y2 - 4), (20, 20, 20), 2)

    return frame


class VisionPieceTests(unittest.TestCase):
    def test_detects_l_pieces_without_expanding_to_phantom_cells(self):
        shapes = [
            np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8),
            np.array([[1, 0, 0], [1, 1, 1]], dtype=np.uint8),
            np.array([[1, 1, 1], [0, 0, 1]], dtype=np.uint8),
        ]

        frame = draw_inventory(shapes)
        detected = vision.get_pieces(frame, 49.0, 49.0)

        self.assertEqual([piece.tolist() for piece in detected], [shape.tolist() for shape in shapes])

    def test_detects_square_and_one_wide_line_pieces(self):
        shapes = [
            np.array([[1, 1], [1, 1]], dtype=np.uint8),
            np.array([[1, 1, 1, 1]], dtype=np.uint8),
            np.array([[1], [1], [1], [1]], dtype=np.uint8),
        ]

        frame = draw_inventory(shapes)
        detected = vision.get_pieces(frame, 49.0, 49.0)

        self.assertEqual([piece.tolist() for piece in detected], [shape.tolist() for shape in shapes])

    def test_detects_five_wide_line_when_inventory_blocks_are_smaller_than_board_scale(self):
        shapes = [
            np.array([[1, 1, 1, 1, 1]], dtype=np.uint8),
            np.array([[1, 1, 1, 1, 1]], dtype=np.uint8),
            np.array([[1, 1, 1, 1, 1]], dtype=np.uint8),
        ]

        frame = draw_inventory(shapes, block_size=15)
        detected = vision.get_pieces(frame, 49.0, 49.0)

        self.assertEqual([piece.tolist() for piece in detected], [shape.tolist() for shape in shapes])

    def test_detects_vertical_line_without_counting_drop_shadow_as_column(self):
        frame_h = 320
        frame_w = 480
        roi_x = 60
        roi_y = 170
        roi_w = 360
        roi_h = 130
        block_size = 20
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        frame[:] = (126, 166, 196)
        config.PIECES_ROI = [roi_x / frame_w, roi_y / frame_h, roi_w / frame_w, roi_h / frame_h]

        slot_w = roi_w // 3
        colors = [(210, 230, 40), (60, 220, 60), (220, 130, 60), (230, 220, 70)]
        for slot in range(3):
            start_x = roi_x + slot * slot_w + (slot_w - block_size) // 2
            start_y = roi_y + (roi_h - 4 * block_size) // 2
            for row, color in enumerate(colors):
                x1 = start_x
                y1 = start_y + row * block_size
                x2 = x1 + block_size - 1
                y2 = y1 + block_size - 1
                cv2.rectangle(frame, (x2 + 1, y1 + 4), (x2 + 18, y2 + 8), (75, 85, 95), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1)

        detected = vision.get_pieces(frame, 38.0, 38.0)

        self.assertEqual(
            [piece.tolist() for piece in detected],
            [[[1], [1], [1], [1]], [[1], [1], [1], [1]], [[1], [1], [1], [1]]],
        )


if __name__ == "__main__":
    unittest.main()