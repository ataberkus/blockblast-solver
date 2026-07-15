"""Measure warm solver latency on a repeatable constrained board."""

import argparse
import statistics
import time

import numpy as np

from block_blast_solver.modules import solver


def benchmark(iterations: int, node_budget: int) -> list[float]:
    board = np.zeros((8, 8), dtype=np.uint8)
    board[0:4, 0:4] = 1
    board[6, 0:6] = 1
    pieces = [
        np.array([[1, 1], [1, 0]], dtype=np.uint8),
        np.ones((1, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.uint8),
    ]

    solver.solve_with_diagnostics(board, pieces, node_budget=node_budget)
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        solver.solve_with_diagnostics(board, pieces, node_budget=node_budget)
        durations.append((time.perf_counter() - started) * 1000.0)
    return durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--node-budget", type=int, default=0)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    durations = benchmark(args.iterations, max(0, args.node_budget))
    sorted_durations = sorted(durations)
    p95_index = min(len(sorted_durations) - 1, int(len(sorted_durations) * 0.95))
    print(
        f"runs={len(durations)} "
        f"median={statistics.median(durations):.1f}ms "
        f"p95={sorted_durations[p95_index]:.1f}ms "
        f"max={max(durations):.1f}ms"
    )


if __name__ == "__main__":
    main()
