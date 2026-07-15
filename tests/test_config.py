import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from block_blast_solver import config


VALID_BOARD_ROI = [0.1, 0.1, 0.6, 0.6]
VALID_PIECES_ROI = [0.1, 0.75, 0.8, 0.2]


class CalibrationValidationTests(unittest.TestCase):
    def tearDown(self):
        config.BOARD_ROI = None
        config.PIECES_ROI = None

    def test_accepts_valid_normalized_rois(self):
        valid, reason = config.validate_calibration(VALID_BOARD_ROI, VALID_PIECES_ROI)

        self.assertTrue(valid)
        self.assertEqual(reason, "")

    def test_rejects_malformed_and_out_of_bounds_rois(self):
        invalid_rois = [
            None,
            [0.1, 0.2, 0.3],
            [True, 0.2, 0.3, 0.4],
            [-0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.0, 0.4],
            [0.8, 0.2, 0.3, 0.4],
            [0.1, float("nan"), 0.3, 0.4],
        ]

        for roi in invalid_rois:
            with self.subTest(roi=roi):
                valid, reason = config.validate_roi(roi)
                self.assertFalse(valid)
                self.assertTrue(reason)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_file = Path(temporary_directory) / "nested" / "calibration.json"
            with mock.patch.object(config, "CALIBRATION_FILE", str(calibration_file)):
                self.assertTrue(config.save_calibration(VALID_BOARD_ROI, VALID_PIECES_ROI))
                config.BOARD_ROI = None
                config.PIECES_ROI = None

                self.assertTrue(config.load_calibration())

        self.assertEqual(config.BOARD_ROI, VALID_BOARD_ROI)
        self.assertEqual(config.PIECES_ROI, VALID_PIECES_ROI)

    def test_invalid_file_clears_stale_calibration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_file = Path(temporary_directory) / "calibration.json"
            calibration_file.write_text(
                json.dumps({"BOARD_ROI": [0.9, 0.1, 0.4, 0.4], "PIECES_ROI": VALID_PIECES_ROI}),
                encoding="utf-8",
            )
            config.BOARD_ROI = VALID_BOARD_ROI
            config.PIECES_ROI = VALID_PIECES_ROI

            with mock.patch.object(config, "CALIBRATION_FILE", str(calibration_file)):
                self.assertFalse(config.load_calibration())

        self.assertIsNone(config.BOARD_ROI)
        self.assertIsNone(config.PIECES_ROI)

    def test_save_rejects_invalid_values_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_file = Path(temporary_directory) / "calibration.json"
            with mock.patch.object(config, "CALIBRATION_FILE", str(calibration_file)):
                self.assertFalse(config.save_calibration([0.0, 0.0, 2.0, 1.0], VALID_PIECES_ROI))

            self.assertFalse(calibration_file.exists())


if __name__ == "__main__":
    unittest.main()
