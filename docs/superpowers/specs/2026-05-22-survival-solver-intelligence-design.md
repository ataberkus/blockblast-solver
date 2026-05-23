# Survival Solver Intelligence Design

**Date:** 2026-05-22  
**Status:** Approved design, awaiting implementation plan

## Goal

Improve the Block Blast solver so it optimizes for long-term survival and higher eventual score, not only immediate clears or short-term board shape.

## Current Context

The current solver in `block_blast_solver/modules/solver.py` already exhaustively searches all legal placements and orders for the currently visible inventory pieces. It uses bitmasks and Numba for speed. After all current pieces are placed, it scores the final board using:

- row/column clear reward
- empty cell reward
- hole penalty
- bumpiness penalty
- readiness bonus for 3x3, 5x1, and 1x5 pieces

The search depth is appropriate because Block Blast only reveals the current set of pieces. The main improvement should be the final-state evaluation function.

## Chosen Approach

Use a future-readiness heuristic. Keep the existing exhaustive current-piece search, but replace the final board evaluation with richer survival scoring.

This avoids a large architecture change, keeps the HUD responsive, and directly targets the user's goal: avoid dead ends and reach higher long-term scores.

## Architecture

The solver remains split into two responsibilities:

1. Search: enumerate all reachable board states after placing the current pieces.
2. Evaluation: rank those reachable states by long-term survivability.

The upgrade is isolated to `block_blast_solver/modules/solver.py`. Capture, vision, calibration, and HUD rendering do not need to change for this feature.

## Scoring Features

### Future Piece Fit Count

Count how many common future piece families can fit anywhere on the resulting board. Reward states that preserve options for:

- single blocks
- 1x2 / 2x1
- 1x3 / 3x1
- 1x4 / 4x1
- 1x5 / 5x1
- 2x2
- 3x3
- common L, T, and Z shapes

This becomes the main survival signal.

### Largest Empty Region

Use flood fill or bitmask connectivity to measure connected empty regions. Reward large open areas and penalize tiny isolated regions.

### Line Readiness

Reward rows and columns that are one or two cells away from clearing, especially when those missing cells are in places common shapes can fill.

### Trap Penalties

Penalize patterns that commonly lead to dead ends:

- isolated single-cell holes
- narrow unreachable pockets
- long skinny gaps that only rare pieces can fill
- jagged boundaries that split the board into disconnected areas

### Immediate Clear Value

Keep row/column clear rewards, but make them less dominant than future-readiness. Immediate clears are valuable when they improve survival, not when they destroy open space.

## Configuration

Expose the new weights in `config.py` so tuning does not require editing JIT logic repeatedly. Candidate weights:

- `W_FUTURE_FITS`
- `W_LARGEST_REGION`
- `W_SMALL_REGION_PENALTY`
- `W_LINE_READINESS`
- `W_TRAP_PENALTY`

Existing weights may remain for compatibility, but the new evaluator should be the primary scoring path.

## Error Handling

If no legal placement sequence exists, `solve()` should continue returning `(None, best_score)` as it does today.

If vision produces missing pieces, the solver should evaluate only active pieces exactly as it currently does.

## Testing

Add deterministic tests or a small test script that bypasses camera input and calls `solver.solve()` directly.

Minimum scenarios:

1. The solver never suggests placing on occupied cells.
2. The solver preserves a 3x3 empty area when no urgent clear is available.
3. The solver chooses a line clear when it prevents a near-term dead end.
4. The solver penalizes isolated single-cell holes.
5. The solver prefers larger connected empty regions over fragmented empty regions.
6. Runtime remains fast enough for live HUD use, ideally under 50 ms after Numba warmup for normal boards.

## Out of Scope

- Vision and camera detection improvements
- Automatic gameplay input or phone control
- Machine learning
- Monte Carlo simulation of unknown future piece sets
- HUD redesign

## Success Criteria

The upgraded solver should produce more conservative, survival-oriented placements. It should avoid filling valuable open space for small immediate score gains and should noticeably reduce dead-end suggestions during live play.
