import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_vision_training_data.py"


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_vision_training_data", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExportVisionTrainingDataTests(unittest.TestCase):
    def test_export_writes_board_and_inventory_pairs(self):
        export_module = _load_export_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_root = root / "fixtures"
            out_dir = root / "dataset"
            fixture_root.mkdir()

            frame = np.full((160, 160, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (16, 16), (31, 27), (200, 80, 40), -1)
            cv2.rectangle(frame, (28, 128), (43, 143), (40, 180, 220), -1)
            cv2.imwrite(str(fixture_root / "tiny.png"), frame)
            label = {
                "board_roi": [0.1, 0.1, 0.8, 0.6],
                "pieces_roi": [0.1, 0.75, 0.8, 0.2],
                "board": [[1 if (r, c) == (0, 0) else 0 for c in range(8)] for r in range(8)],
                "pieces": [[[1]], None, None],
            }
            (fixture_root / "tiny.json").write_text(json.dumps(label), encoding="utf-8")

            manifest = export_module.export_training_data(out_dir, fixture_root)

            self.assertEqual(manifest["board_cells_occupied"], 1)
            self.assertEqual(manifest["board_cells_empty"], 63)
            self.assertEqual(manifest["inventory_pairs"], 1)
            self.assertTrue((out_dir / "board_cells" / "occupied" / "tiny_r0_c0.png").exists())
            self.assertTrue((out_dir / "inventory" / "images" / "tiny_slot0.png").exists())
            self.assertTrue((out_dir / "inventory" / "masks" / "tiny_slot0.png").exists())
            mask = cv2.imread(str(out_dir / "inventory" / "masks" / "tiny_slot0.png"), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(mask)
            self.assertGreater(int(np.count_nonzero(mask)), 0)


if __name__ == "__main__":
    unittest.main()
