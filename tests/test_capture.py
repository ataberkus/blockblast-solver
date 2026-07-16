import unittest

import numpy as np

from block_blast_solver import config
from block_blast_solver.modules.capture import StaticFrameCapture, WindowCapture, roi_to_rect


class FakeCamera:
    def __init__(self, fail_release=False):
        self.fail_release = fail_release
        self.calls = []

    def release(self):
        self.calls.append("release")
        if self.fail_release:
            raise RuntimeError("release failed")

    def stop(self):
        self.calls.append("stop")


class StaticCaptureTests(unittest.TestCase):
    def test_returns_independent_frame_copies(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        capture = StaticFrameCapture(frame)

        first = capture.capture_frame()
        first[0, 0] = 255
        second = capture.capture_frame()

        self.assertEqual(int(np.sum(second[0, 0])), 0)

    def test_maps_valid_normalized_roi(self):
        rect = roi_to_rect([0.1, 0.2, 0.5, 0.4], (100, 200, 3))

        self.assertEqual(rect, (20, 20, 100, 40))

    def test_rejects_invalid_roi(self):
        with self.assertRaises(ValueError):
            roi_to_rect([0.8, 0.2, 0.5, 0.4], (100, 200, 3))

    def test_calibration_status_uses_shared_config(self):
        previous_board = config.BOARD_ROI
        previous_pieces = config.PIECES_ROI
        try:
            capture = StaticFrameCapture(np.zeros((10, 10, 3), dtype=np.uint8))
            config.BOARD_ROI = [0.0, 0.0, 0.5, 0.5]
            config.PIECES_ROI = [0.0, 0.5, 1.0, 0.5]
            self.assertTrue(capture.calibrate())
        finally:
            config.BOARD_ROI = previous_board
            config.PIECES_ROI = previous_pieces

    def test_window_capture_close_releases_cameras_without_raising(self):
        capture = object.__new__(WindowCapture)
        failing_camera = FakeCamera(fail_release=True)
        ok_camera = FakeCamera()
        capture.cameras = {0: failing_camera, 1: ok_camera}
        capture.monitor_rects = {0: (0, 0, 100, 100), 1: (100, 0, 200, 100)}
        capture.window_handle = object()

        capture.close()

        self.assertEqual(failing_camera.calls, ["stop", "release"])
        self.assertEqual(ok_camera.calls, ["stop", "release"])
        self.assertEqual(capture.cameras, {})
        self.assertEqual(capture.monitor_rects, {})
        self.assertIsNone(capture.window_handle)


if __name__ == "__main__":
    unittest.main()
