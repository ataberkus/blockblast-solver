import unittest

from block_blast_solver import config
from block_blast_solver.modules import vision
from block_blast_solver.modules.vision_fixtures import iter_skin_fixtures
from block_blast_solver.modules.vision_models import ModelRegistry


class VisionSkinRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ModelRegistry.reset_for_tests()
        registry = ModelRegistry.get()
        if not registry.using_learned:
            raise unittest.SkipTest("ONNX vision models not available")

    def tearDown(self):
        ModelRegistry.reset_for_tests()

    def test_each_skin_matches_labels(self):
        for fixture in iter_skin_fixtures():
            with self.subTest(fixture=fixture.name):
                previous_board = config.BOARD_ROI
                previous_pieces = config.PIECES_ROI
                try:
                    config.BOARD_ROI = fixture.board_roi
                    config.PIECES_ROI = fixture.pieces_roi
                    board, occluded = vision.get_board_state(fixture.image)
                    self.assertFalse(occluded, fixture.name)
                    self.assertEqual(board.tolist(), fixture.board.tolist(), fixture.name)
                    cell_w = fixture.image.shape[1] * fixture.board_roi[2] / 8.0
                    cell_h = fixture.image.shape[0] * fixture.board_roi[3] / 8.0
                    pieces = vision.get_pieces(fixture.image, cell_w, cell_h)
                    for got, expected in zip(pieces, fixture.pieces):
                        if expected is None:
                            self.assertIsNone(got, fixture.name)
                        else:
                            self.assertIsNotNone(got, fixture.name)
                            self.assertEqual(got.tolist(), expected.tolist(), fixture.name)
                finally:
                    config.BOARD_ROI = previous_board
                    config.PIECES_ROI = previous_pieces


if __name__ == "__main__":
    unittest.main()
