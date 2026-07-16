import importlib
import os
import unittest
from unittest import mock

import numpy as np

from block_blast_solver import config
from block_blast_solver.modules import vision_models


class VisionModelConfigTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("BLOCK_BLAST_BOARD_CELL_MODEL", None)
        os.environ.pop("BLOCK_BLAST_INVENTORY_MASK_MODEL", None)
        os.environ.pop("VISION_FORCE_HEURISTIC", None)
        importlib.reload(config)

    def test_exports_default_model_paths_and_thresholds(self):
        self.assertTrue(config.BOARD_CELL_MODEL_PATH.endswith("models/board_cell_classifier.onnx"))
        self.assertTrue(config.INVENTORY_MASK_MODEL_PATH.endswith("models/inventory_slot_masker.onnx"))
        self.assertEqual(config.T_BOARD, 0.5)
        self.assertEqual(config.T_MASK, 0.5)
        self.assertEqual(config.T_CELL, 0.30)
        self.assertFalse(config.VISION_FORCE_HEURISTIC)
        self.assertFalse(config.vision_force_heuristic())

    def test_reads_model_path_and_force_heuristic_overrides_from_env(self):
        os.environ["BLOCK_BLAST_BOARD_CELL_MODEL"] = "/tmp/board.onnx"
        os.environ["BLOCK_BLAST_INVENTORY_MASK_MODEL"] = "/tmp/mask.onnx"
        os.environ["VISION_FORCE_HEURISTIC"] = "true"

        reloaded = importlib.reload(config)

        self.assertEqual(reloaded.BOARD_CELL_MODEL_PATH, "/tmp/board.onnx")
        self.assertEqual(reloaded.INVENTORY_MASK_MODEL_PATH, "/tmp/mask.onnx")
        self.assertTrue(reloaded.VISION_FORCE_HEURISTIC)
        self.assertTrue(reloaded.vision_force_heuristic())


class VisionModelsTests(unittest.TestCase):
    def setUp(self):
        vision_models.ModelRegistry.reset_for_tests()

    def tearDown(self):
        vision_models.ModelRegistry.reset_for_tests()
        os.environ.pop("VISION_FORCE_HEURISTIC", None)

    def test_force_heuristic_disables_models(self):
        os.environ["VISION_FORCE_HEURISTIC"] = "1"
        vision_models.ModelRegistry.reset_for_tests()

        registry = vision_models.ModelRegistry.get()

        self.assertFalse(registry.using_learned)
        self.assertIsNone(registry.board_classifier)
        self.assertIsNone(registry.inventory_masker)

    def test_missing_weights_fall_back(self):
        with self.assertLogs(vision_models.logger, level="WARNING") as captured:
            with mock.patch.object(vision_models, "DEFAULT_BOARD_MODEL", "/no/such/board.onnx"), mock.patch.object(
                vision_models, "DEFAULT_MASK_MODEL", "/no/such/mask.onnx"
            ):
                vision_models.ModelRegistry.reset_for_tests()
                registry = vision_models.ModelRegistry.get()

        self.assertFalse(registry.using_learned)
        self.assertIsNone(registry.board_classifier)
        self.assertIsNone(registry.inventory_masker)
        self.assertEqual(len(captured.output), 2)

    def test_occlusion_helper_flags_uncertain_board(self):
        probs = np.full((8, 8), 0.5, dtype=np.float32)

        self.assertTrue(vision_models.board_probs_are_occluded(probs))

        empty = np.full((8, 8), 0.05, dtype=np.float32)
        self.assertFalse(vision_models.board_probs_are_occluded(empty))


if __name__ == "__main__":
    unittest.main()
