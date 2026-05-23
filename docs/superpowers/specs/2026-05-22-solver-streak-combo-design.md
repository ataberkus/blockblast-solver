# Solver Streak / Combo Intelligence — Design Spec

**Date:** 2026-05-22
**Status:** Approved for planning
**Related work:** builds on `2026-05-22-monte-carlo-survival-solver-design.md`

## Problem

Block Blast rewards a *combo counter* that grows on every consecutive
placement that clears at least one row or column. High combos (e.g. 50+)
yield massive point bonuses. Today our solver:

- Scores every cleared line linearly via `W_CLEAR=500`, with no extra
  reward for keeping the combo streak alive across placements.
- Has no penalty for a zero-clear placement that breaks the streak.
- Optimizes Monte Carlo for *survival* of the next set, not for the next
  set's ability to *continue clearing*.

User goal: maximize **consistency of clears** — clear-after-clear-after-clear
matters far more than how many lines any single placement clears.

## Goals

1. Reward every placement (across the current 3-piece set and within the
   simulated next set) that produces at least one clear.
2. Strongly penalize any placement in the candidate 3-piece plan that
   produces zero clears.
3. Prefer board states where the *next* random set is also likely to
   continue the streak.
4. Keep warm solver runtime within the existing ~100 ms live-HUD budget.

## Non-Goals (YAGNI)

- Reading the on-screen combo counter via vision.
- Geometric / multiplicative combo scoring (Approach B in brainstorming).
- Search depth beyond the current 3 pieces.
- Modeling the game's exact internal point formula.

## Design

### Section 1 — Per-placement streak scoring in `solve_recursive`

**File:** `block_blast_solver/modules/solver.py`

Inside the 3-piece recursion (currently lines ~594–710), in addition to
the existing `accumulated_score + ((cleared_rows + cleared_cols) * w_clear)`
update, we track a streak counter that flows through the recursion:

- `streak_in_plan` (int): starts at 0 when recursion begins.
- For each placement:
  - `is_clearing = (cleared_rows + cleared_cols) > 0`
  - If `is_clearing`: `streak_in_plan += 1`, add `W_STREAK_CONTINUE` to
    accumulated score.
  - Else: `streak_in_plan = 0`, add `W_STREAK_BREAK_PENALTY` (negative)
    to accumulated score.
- At the leaf (after the 3rd placement): if `streak_in_plan == 3`, add
  `W_STREAK_PERFECT_BONUS` once.

**New config weights** (`block_blast_solver/config.py`):

```python
W_STREAK_CONTINUE        =  250.0
W_STREAK_BREAK_PENALTY   = -900.0
W_STREAK_PERFECT_BONUS   =  600.0
```

Rationale: `W_CLEAR=500` per cleared line means a 2-line single
placement gives +1000. The break penalty (−900) must outweigh the
temptation to bank a single fat clear and then break the streak on the
next placement. The perfect bonus rewards the solver for finding plans
where all three placements clear.

All three new weights flow into the existing JIT signature alongside
`w_clear`, so disabling streak scoring requires only setting these
weights to 0.0 in config.

### Section 2 — Monte Carlo next-set streak continuity

**File:** `block_blast_solver/modules/solver.py`

Extend `simulate_next_set_survival_jit` to also return
`clearing_placements` (int): the number of placed pieces in the
simulated next set that produced ≥1 clear.

Update `evaluate_monte_carlo_survival_jit` to compute one new metric
across samples whose simulation placed at least one piece
(`placed_count > 0`):

```
avg_streak_continuity = mean(clearing_placements / placed_count)
                        over samples with placed_count > 0
```

If zero samples placed any piece (board already effectively dead),
`avg_streak_continuity = 0.0`.

This represents the expected fraction of next-set placements that will
keep the streak alive, given we land on the candidate board state.

**New config weight:**

```python
W_MONTE_CARLO_STREAK = 200.0   # added to terminal score, scaled by avg_streak_continuity ∈ [0,1]
```

Sample budget unchanged: 2 normal / 4 danger. The metric reuses
`simulate_best_single_future_piece_jit` so no additional simulation cost
is incurred.

**Diagnostics:** add `next_streak_pct ∈ [0.0, 1.0]` to the diagnostics
dict returned by `_solve_core` / `solve_with_diagnostics`.

### Section 3 — HUD field

**File:** `block_blast_solver/modules/visualizer.py`

Extend `draw_hud` to render a new line under the existing Monte Carlo
fields:

```
Next streak: 75%
```

Format: `"Next streak: {int(next_streak_pct * 100)}%"`. Skip the line if
the key is missing (backwards compatible).

## Architecture / Data Flow

```
main.py
  -> solver.solve_with_diagnostics(board, pieces)
       -> _solve_core(board, active_pieces)
            -> solve_recursive(...)                  # Section 1 logic here
                 -> evaluate_survival_score_jit(...)
                      -> evaluate_monte_carlo_survival_jit(...)
                           -> simulate_next_set_survival_jit(...)   # Section 2 logic
       -> diagnostics dict { ..., next_streak_pct }
  -> visualizer.draw_hud(..., diagnostics=...)       # Section 3 renders new field
```

No new modules, no new top-level functions. Pure additive changes.

## Testing

New file: `tests/test_solver_streak.py`

1. `test_break_penalty_dominates_single_fat_clear` — given two plans:
   Plan A clears 3 lines on piece 1 then 0+0; Plan B clears 1+1+1.
   Solver must choose Plan B.
2. `test_perfect_streak_bonus_awarded` — verify the +600 bonus appears
   only when all 3 placements clear.
3. `test_streak_reset_on_break` — placements clear/zero/clear: bonuses
   accrue on placements 1 and 3 only; no perfect bonus.
4. `test_mc_streak_continuity_metric` — craft a board where the greedy
   next-set sim clears on most placements; assert
   `avg_streak_continuity > 0.6`. Inverse craft asserts `< 0.3`.
5. `test_solver_with_diagnostics_returns_next_streak_pct` — diagnostics
   dict contains key `next_streak_pct` with value in `[0.0, 1.0]`.

HUD test (extend `tests/test_visualizer.py` if present, else new file
`tests/test_visualizer_hud.py`):

6. `test_draw_hud_renders_next_streak_field` — when diagnostics
   contains `next_streak_pct=0.75`, the rendered output includes
   `"Next streak: 75%"`.

All existing 18 tests must continue to pass unchanged.

## Rollout

- All new weights live in `config.py`; setting them to 0.0 reproduces
  pre-change behavior.
- After implementation, run live capture for at least one game session
  and confirm the HUD shows long streaks and that point totals improve
  vs. the prior session baseline.
- Performance gate: warm solver runtime must stay ≤ 150 ms per move
  (current ~100 ms). Verify by re-running existing timing harness.

## Risks

- **Weight imbalance:** if `W_STREAK_BREAK_PENALTY` is too small, solver
  still trades streaks for fat single clears. If too large, it may
  accept survival-damaging plans just to keep the streak. Mitigation:
  tunable via config; first live session is the validation.
- **MC noise:** with only 2 samples normally, `avg_streak_continuity`
  has high variance. We accept this because (a) it averages out across
  many moves and (b) the metric is one component among several. No
  sample budget increase.
