import unittest

import numpy as np

from block_blast_solver.modules import solver


def mask_from_board(board):
    mask = 0
    for row in range(8):
        for col in range(8):
            if board[row, col]:
                mask |= 1 << (row * 8 + col)
    return np.int64(np.uint64(mask))


class FutureSetEvaluationTests(unittest.TestCase):
    def test_catalog_family_weights_have_equal_total_mass(self):
        _, _, families = solver.get_monte_carlo_piece_catalog()
        cumulative, _ = solver.get_catalog_cumulative_weights(families)
        weights = np.diff(np.concatenate((np.array([0], dtype=np.int32), cumulative)))

        family_totals = {}
        for family, weight in zip(families, weights):
            family_totals.setdefault(int(family), 0)
            family_totals[int(family)] += int(weight)

        self.assertEqual(len(set(family_totals.values())), 1)

    def test_permutation_aware_simulation_finds_surviving_order(self):
        board = np.ones((8, 8), dtype=np.uint8)
        board[0, :] = 0
        board[1, 0] = 0
        board_mask = mask_from_board(board)

        catalog_masks = np.array([0x303, 0x1], dtype=np.uint64)
        catalog_shapes = np.array([[2, 2], [1, 1]], dtype=np.int32)
        sample = np.array([0, 1, 1], dtype=np.int32)

        fixed_result = solver.simulate_next_set_one_order_jit(
            board_mask,
            sample,
            catalog_masks,
            catalog_shapes,
        )
        permutation_result = solver.simulate_next_set_survival_jit(
            board_mask,
            sample,
            catalog_masks,
            catalog_shapes,
        )

        self.assertFalse(fixed_result[0])
        self.assertTrue(permutation_result[0])
        self.assertEqual(permutation_result[3], 3)

    def test_same_seed_replays_exact_future_sets(self):
        first = solver.get_next_piece_samples(100, board_mask=0x1234, seed=99)
        replay = solver.get_next_piece_samples(100, board_mask=0x1234, seed=99)
        other_seed = solver.get_next_piece_samples(100, board_mask=0x1234, seed=100)

        self.assertTrue(np.array_equal(first, replay))
        self.assertFalse(np.array_equal(first, other_seed))


if __name__ == "__main__":
    unittest.main()
