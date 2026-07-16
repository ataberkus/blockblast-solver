import unittest
from unittest import mock

import numpy as np

from block_blast_solver import config
from block_blast_solver.modules.visualizer import Visualizer, summarize_move_sequence


class VisualizerOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.previous_board_roi = config.BOARD_ROI
        self.previous_pieces_roi = config.PIECES_ROI

    def tearDown(self):
        config.BOARD_ROI = self.previous_board_roi
        config.PIECES_ROI = self.previous_pieces_roi

    def test_summarizes_multi_step_column_clear(self):
        board = np.zeros((8, 8), dtype=np.uint8)
        board[7, 0:3] = 1
        pieces = [
            np.ones((1, 5), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
        ]
        moves = [
            {"slot_index": 0, "row": 6, "col": 0},
            {"slot_index": 1, "row": 0, "col": 0},
            {"slot_index": 2, "row": 3, "col": 0},
        ]

        outcomes, total_clears, final_filled = summarize_move_sequence(board, pieces, moves)

        self.assertEqual([outcome["clear_count"] for outcome in outcomes], [0, 0, 3])
        self.assertEqual(outcomes[2]["cleared_cols"], [0, 1, 2])
        self.assertEqual(total_clears, 3)
        self.assertEqual(final_filled, 2)

    def test_draw_hud_places_panel_outside_game_frame(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [None, None, None]

        hud = Visualizer().draw_hud(frame, board, pieces, None, 0.0, False)

        self.assertEqual(hud.shape[0], frame.shape[0])
        self.assertEqual(hud.shape[1], frame.shape[1] + 320)

    def test_summary_marks_overlapping_move_invalid_without_mutating_board(self):
        board = np.zeros((8, 8), dtype=np.uint8)
        board[0, 0] = 1
        pieces = [np.ones((1, 2), dtype=np.uint8)]
        moves = [{"slot_index": 0, "row": 0, "col": 0}]

        outcomes, total_clears, final_filled = summarize_move_sequence(board, pieces, moves)

        self.assertTrue(outcomes[0]["invalid"])
        self.assertEqual(total_clears, 0)
        self.assertEqual(final_filled, 1)

    def test_draw_hud_accepts_solver_diagnostics(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [None, None, None]
        diagnostics = {
            "risk_level": "high",
            "next_survival_pct": 28.5,
            "clear_routes": 2,
            "future_fits": 9,
            "sample_count": 64,
        }

        hud = Visualizer().draw_hud(frame, board, pieces, None, 0.0, False, diagnostics)

        self.assertEqual(hud.shape[0], frame.shape[0])
        self.assertEqual(hud.shape[1], frame.shape[1] + 320)

    def test_draw_hud_renders_next_streak_line(self):
        frame = np.zeros((400, 300, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [None, None, None]
        diagnostics = {
            "risk_level": "low",
            "next_survival_pct": 80.0,
            "clear_routes": 4,
            "future_fits": 12,
            "sample_count": 64,
            "next_streak_pct": 0.75,
        }

        rendered_texts = []
        from block_blast_solver.modules import visualizer as viz_module
        real_put_text = viz_module.cv2.putText

        def capture(img, text, org, *args, **kwargs):
            rendered_texts.append(text)
            return real_put_text(img, text, org, *args, **kwargs)

        with mock.patch.object(viz_module.cv2, "putText", side_effect=capture):
            Visualizer().draw_hud(frame, board, pieces, None, 0.0, False, diagnostics)

        self.assertTrue(any("Next streak: 75%" in t for t in rendered_texts),
                        f"Expected 'Next streak: 75%' in HUD texts, got: {rendered_texts}")

    def test_draw_hud_renders_move_overlays_with_single_blend(self):
        config.BOARD_ROI = [0.1, 0.1, 0.8, 0.8]
        config.PIECES_ROI = [0.1, 0.75, 0.8, 0.2]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [np.ones((1, 2), dtype=np.uint8), None, None]
        moves = [{"slot_index": 0, "row": 1, "col": 2}]

        from block_blast_solver.modules import visualizer as viz_module
        blend_calls = []
        real_add_weighted = viz_module.cv2.addWeighted
        rendered_texts = []
        real_put_text = viz_module.cv2.putText

        def capture_blend(*args, **kwargs):
            blend_calls.append(1)
            return real_add_weighted(*args, **kwargs)

        def capture_text(img, text, org, *args, **kwargs):
            rendered_texts.append(text)
            return real_put_text(img, text, org, *args, **kwargs)

        with mock.patch.object(viz_module.cv2, "addWeighted", side_effect=capture_blend):
            with mock.patch.object(viz_module.cv2, "putText", side_effect=capture_text):
                hud = Visualizer().draw_hud(frame, board, pieces, moves, 42.5, False)

        self.assertEqual(hud.shape, (240, 640, 3))
        self.assertEqual(len(blend_calls), 1)
        self.assertTrue(any("1. P1 -> (1,2)" in text for text in rendered_texts))
        self.assertTrue(any("Estimated score: 42.5" in text for text in rendered_texts))

    def test_draw_hud_skips_malformed_moves_without_crashing(self):
        config.BOARD_ROI = [0.1, 0.1, 0.8, 0.8]
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [np.ones((1, 1), dtype=np.uint8), None, None]
        moves = [
            {"row": 0, "col": 0},
            {"slot_index": 99, "row": 0, "col": 0},
            {"slot_index": 1, "row": 0, "col": 0},
            {"slot_index": 0, "row": None, "col": 0},
            {"slot_index": 0, "row": 0, "col": "bad"},
            {"slot_index": 0, "row": 8, "col": 0},
        ]

        hud = Visualizer().draw_hud(frame, board, pieces, moves, 0.0, False)

        self.assertEqual(hud.shape, (240, 640, 3))

    def test_draw_hud_shows_occluded_and_warning_status_messages(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [np.ones((1, 1), dtype=np.uint8), None, None]

        from block_blast_solver.modules import visualizer as viz_module
        rendered_texts = []
        real_put_text = viz_module.cv2.putText

        def capture_text(img, text, org, *args, **kwargs):
            rendered_texts.append(text)
            return real_put_text(img, text, org, *args, **kwargs)

        with mock.patch.object(viz_module.cv2, "putText", side_effect=capture_text):
            Visualizer().draw_hud(frame, board, pieces, None, 0.0, True)
            Visualizer().draw_hud(frame, board, pieces, None, 0.0, False)

        self.assertTrue(any(text == "OCCLUDED" for text in rendered_texts))
        self.assertTrue(any("Occluded; using last advice" in text for text in rendered_texts))
        self.assertTrue(any("WARNING: No valid move" in text for text in rendered_texts))


if __name__ == "__main__":
    unittest.main()
