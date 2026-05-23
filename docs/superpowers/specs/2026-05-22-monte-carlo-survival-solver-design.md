# Monte Carlo Survival Solver Design

**Date:** 2026-05-22
**Status:** Self-reviewed, ready for user review

## Goal

Reduce late-game dead ends where the solver reaches a board state that cannot accept the next piece set. The solver should prefer moves that maximize the chance of surviving the next unknown set of pieces, especially on crowded boards like the current screenshot.

## Current Context

The current solver exhaustively searches every legal order and placement for the visible inventory pieces. After all current pieces are placed, it scores the final board with heuristic signals such as immediate clears, empty cells, connected regions, future fit counts, line readiness, and trap penalties.

That is strong for the current inventory, but it still only evaluates one board state at the end of the visible pieces. It does not simulate likely next piece sets, so it can choose a board that looks acceptable now but has poor survival odds after the next draw.

## Chosen Approach

Add bounded Monte Carlo future simulation to the terminal evaluator.

For every fully placed current-piece sequence, the solver will simulate a fixed budget of likely next 3-piece sets from a curated piece catalog. Each simulated next set checks whether the resulting board can place pieces, create clears, and remain open enough for follow-up pieces.

This keeps the exact current-piece search and adds future awareness only at the scoring stage.

## Scoring Priorities

Monte Carlo scoring should use this priority order:

1. **Survival rate:** Highest priority. Prefer terminal boards where the largest percentage of sampled next piece sets can be played.
2. **Future clear routes:** Main tie-breaker. Prefer boards where sampled next pieces can clear rows or columns soon.
3. **Open-space preservation:** Prefer large connected empty regions and available 3x3 / 5-line spaces.
4. **Immediate clears:** Still valuable, but not dominant unless they improve future survival.

This means the solver should not chase a current clear if it leaves the next draw nearly unplayable.

## Piece Catalog

Use a curated catalog of common Block Blast piece families:

- 1x1
- 1x2 and 2x1
- 1x3 and 3x1
- 1x4 and 4x1
- 1x5 and 5x1
- 2x2 square
- 3x3 square
- L pieces and mirrored/rotated variants
- T pieces and rotated variants
- Z/S pieces and rotated variants
- small corner/step shapes when verified from live detection

The catalog should be deterministic and stored in solver code as masks/shapes so Numba can evaluate it quickly.

## Monte Carlo Budget

Use deterministic pseudo-random or precomputed samples rather than uncontrolled randomness.

Initial budget:

- Normal mode: evaluate 2 deterministic next-set samples to keep HUD responsive.
- Danger mode: evaluate 4 deterministic next-set samples when the board is crowded or future fit count is low.
- The helper functions support larger sample counts for tests and tuning, but live HUD defaults must stay small unless runtime is remeasured.
- Runtime target: keep warm solver calls under 250 ms on typical live boards. If that target is missed, reduce sample counts before changing solver behavior.

Danger mode activates when one or more are true:

- filled cells exceed a configured threshold
- future piece fit score falls below a configured threshold
- no 3x3 space remains
- no 1x5 or 5x1 space remains
- empty regions are fragmented
- current board has very few near-clear rows/columns

## Architecture

Keep the public API unchanged:

```python
solve(board, active_pieces) -> (moves, best_score)
```

Add internal helpers in `block_blast_solver/modules/solver.py`:

- `get_monte_carlo_piece_catalog()`
- `get_next_piece_samples()`
- `simulate_next_set_survival_jit()`
- `evaluate_monte_carlo_survival_jit()`
- danger mode threshold helpers

Add configuration values in `block_blast_solver/config.py`:

- `MONTE_CARLO_NORMAL_SAMPLES = 2`
- `MONTE_CARLO_DANGER_SAMPLES = 4`
- `MONTE_CARLO_DANGER_FILLED_CELLS = 38`
- `MONTE_CARLO_MIN_FUTURE_FITS = 8`

The existing recursive search continues to generate all legal current-piece sequences. At terminal depth, the evaluator combines current survival heuristics with Monte Carlo survival results.

## HUD Feedback

Extend HUD text so a surprising move is explainable. Example fields:

- `Risk: High / Medium / Low`
- `Next survival: 68%`
- `Clear routes: 4`
- `Future fits: 21`

This should make it clear when the solver chooses a move because it improves next-set survival instead of immediate board appearance.

## Error Handling

If no complete current-piece sequence exists, keep returning `(None, best_score)` for now. Partial-sequence fallback is useful but should be a separate design because it changes gameplay behavior.

If Monte Carlo samples cannot be evaluated due to malformed piece masks, skip the malformed sample and continue scoring the remaining samples.

If runtime exceeds the configured budget, fall back to the existing survival evaluator for that frame rather than freezing the HUD.

## Testing

Add deterministic tests that bypass camera input and call solver helpers directly.

Minimum tests:

1. Crowded-board state prefers the move with higher next-set survival rate.
2. Immediate clear loses when it creates a low-survival dead-end board.
3. Immediate clear wins when it materially increases survival rate.
4. Future clear routes are rewarded when survival rates are tied.
5. Large-piece availability is preserved when possible.
6. Solver still returns `None` when no current piece can legally move.
7. Monte Carlo sample generation is deterministic.
8. Runtime remains acceptable after Numba warmup.

## Out of Scope

- Automatic tapping or phone control
- Vision/calibration changes
- Machine learning
- Full UI redesign
- Partial move fallback when not all current pieces fit

## Success Criteria

The solver should noticeably reduce late-game lockups by choosing moves that leave the board playable for likely future piece sets. On crowded boards, it should prefer survival probability and future clear routes over cosmetic immediate clears.
