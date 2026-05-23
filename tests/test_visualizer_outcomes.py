import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER_ROOT = os.path.join(ROOT, "block_blast_solver")
if SOLVER_ROOT not in sys.path:
    sys.path.insert(0, SOLVER_ROOT)

from modules.visualizer import Visualizer, summarize_move_sequence


class VisualizerOutcomeTests(unittest.TestCase):
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
        import modules.visualizer as viz_module
        real_put_text = viz_module.cv2.putText

        def capture(img, text, org, *args, **kwargs):
            rendered_texts.append(text)
            return real_put_text(img, text, org, *args, **kwargs)

        with mock.patch.object(viz_module.cv2, "putText", side_effect=capture):
            Visualizer().draw_hud(frame, board, pieces, None, 0.0, False, diagnostics)

        self.assertTrue(any("Next streak: 75%" in t for t in rendered_texts),
                        f"Expected 'Next streak: 75%' in HUD texts, got: {rendered_texts}")


if __name__ == "__main__":
    unittest.main()