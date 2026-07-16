import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from block_blast_solver.modules.vision_fixtures import (
    iter_skin_fixtures,
    load_skin_fixture,
    validate_skin_label,
)


class VisionFixtureSchemaTests(unittest.TestCase):
    def test_iter_skin_fixtures_loads_expected_repo_fixture_names(self):
        self.assertEqual(
            [fixture.name for fixture in iter_skin_fixtures()],
            [
                "cake_wafer",
                "gem_busy",
                "gem_five_line",
                "gem_two_rows",
                "heart_pastel",
                "pearl_mixed",
                "stone_corner",
                "watermelon_green",
                "watermelon_purple",
                "wood_classic",
            ],
        )

    def test_validate_accepts_minimal_label(self):
        label = {
            "board_roi": [0.1, 0.2, 0.8, 0.5],
            "pieces_roi": [0.1, 0.75, 0.8, 0.2],
            "board": [[0] * 8 for _ in range(8)],
            "pieces": [None, [[1, 1], [1, 1]], None],
        }
        validate_skin_label(label)

    def test_validate_rejects_wrong_board_shape(self):
        label = {
            "board_roi": [0.1, 0.2, 0.8, 0.5],
            "pieces_roi": [0.1, 0.75, 0.8, 0.2],
            "board": [[0] * 7 for _ in range(8)],
            "pieces": [None, None, None],
        }
        with self.assertRaises(ValueError):
            validate_skin_label(label)

    def test_load_skin_fixture_round_trip(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "demo.png"
            label_path = root / "demo.json"
            frame = np.zeros((200, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), frame)
            label = {
                "board_roi": [0.1, 0.1, 0.8, 0.6],
                "pieces_roi": [0.1, 0.75, 0.8, 0.2],
                "board": [[0] * 8 for _ in range(8)],
                "pieces": [[[1]], None, None],
            }
            label_path.write_text(json.dumps(label), encoding="utf-8")
            fixture = load_skin_fixture(label_path)
            self.assertEqual(fixture.image.shape, (200, 100, 3))
            self.assertEqual(fixture.board.shape, (8, 8))
            self.assertEqual(fixture.pieces[0].tolist(), [[1]])
            self.assertIsNone(fixture.pieces[1])


if __name__ == "__main__":
    unittest.main()
