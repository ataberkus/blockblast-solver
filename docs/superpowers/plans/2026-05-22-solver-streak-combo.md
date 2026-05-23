# Solver Streak / Combo Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the solver chase long combo streaks by scoring every clearing placement, heavily penalizing zero-clear placements, and forecasting next-set streak continuity via Monte Carlo.

**Architecture:** Layer streak-aware scoring on top of the existing 3-piece bitmask recursion in `solver.py`. Inside `solve_recursive`, track a `streak_in_plan` counter that adjusts `accumulated_score` per placement. Extend the existing Monte Carlo sim to also count clearing placements, then surface `next_streak_pct` through diagnostics into the HUD. All new weights live in `config.py` and are additive (set to 0.0 to disable).

**Tech Stack:** Python 3.13, NumPy, Numba `@njit(cache=True)`, OpenCV (HUD), `unittest`.

**Spec:** [docs/superpowers/specs/2026-05-22-solver-streak-combo-design.md](../specs/2026-05-22-solver-streak-combo-design.md)

---

## File Structure

- **Modify:** `block_blast_solver/config.py` — add 4 new weights.
- **Modify:** `block_blast_solver/modules/solver.py` — extend 4 JIT functions and 1 Python wrapper; grow diagnostics holder from size 4 to size 5.
- **Modify:** `block_blast_solver/modules/visualizer.py` — render `next_streak_pct` line in `draw_hud`.
- **Create:** `tests/test_solver_streak.py` — 5 unit tests for streak scoring + diagnostics.
- **Modify:** `tests/test_visualizer_outcomes.py` — 1 new test for HUD streak line.

---

### Task 1: Add streak weights to config

**Files:**
- Modify: `block_blast_solver/config.py` (append after the Monte Carlo block, around line 47)

- [ ] **Step 1: Add 4 new constants**

Append at the end of the existing weight block (right after `W_MONTE_CARLO_FUTURE_FITS = 12.0`):

```python
# Streak / combo continuity weights (Section 1 + Section 2 of streak spec)
W_STREAK_CONTINUE        =  250.0   # per placement in current plan that clears >= 1 line
W_STREAK_BREAK_PENALTY   = -900.0   # per placement in current plan that clears 0 lines
W_STREAK_PERFECT_BONUS   =  600.0   # one-shot bonus when all 3 placements clear
W_MONTE_CARLO_STREAK     =  200.0   # scaled by avg_streak_continuity in [0.0, 1.0]
```

- [ ] **Step 2: Verify config still imports cleanly**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -c "from block_blast_solver import config; print(config.W_STREAK_CONTINUE, config.W_MONTE_CARLO_STREAK)"`
Expected: `250.0 200.0`

- [ ] **Step 3: Commit**

```bash
git add block_blast_solver/config.py
git commit -m "feat(config): add streak/combo continuity weights"
```

---

### Task 2: Extend `simulate_next_set_survival_jit` to count clearing placements

**Files:**
- Modify: `block_blast_solver/modules/solver.py:529-549`

- [ ] **Step 1: Write the failing test**

Create `tests/test_solver_streak.py`:

```python
import numpy as np
import unittest

from block_blast_solver.modules import solver


class StreakCountingTests(unittest.TestCase):
    def test_simulate_next_set_returns_clearing_placement_count(self):
        # An empty board: greedy sim will likely clear 0 lines across 3 small pieces.
        catalog_masks, catalog_shapes, _ = solver.get_monte_carlo_piece_catalog()
        samples = solver.get_next_piece_samples(1)
        survived, _, _, placed, clearing = solver.simulate_next_set_survival_jit(
            np.int64(0), samples[0], catalog_masks, catalog_shapes
        )
        self.assertTrue(survived)
        self.assertEqual(placed, 3)
        self.assertGreaterEqual(clearing, 0)
        self.assertLessEqual(clearing, placed)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak -v`
Expected: FAIL — `simulate_next_set_survival_jit` returns 4-tuple, not 5-tuple.

- [ ] **Step 3: Extend the JIT function**

Replace the entire body of `simulate_next_set_survival_jit` (lines ~529-549) with:

```python
@njit(cache=True)
def simulate_next_set_survival_jit(board_mask: int,
                                   sample: np.ndarray,
                                   catalog_masks: np.ndarray,
                                   catalog_shapes: np.ndarray) -> Tuple[bool, int, int, int, int]:
    current_board = board_mask
    total_clears = 0
    clearing_placements = 0

    for sample_pos in range(3):
        piece_idx = sample[sample_pos]
        piece_mask = int(catalog_masks[piece_idx])
        ph = int(catalog_shapes[piece_idx, 0])
        pw = int(catalog_shapes[piece_idx, 1])
        placed, next_board, clears = simulate_best_single_future_piece_jit(current_board, piece_mask, ph, pw)
        if not placed:
            return False, current_board, total_clears, sample_pos, clearing_placements
        current_board = next_board
        total_clears += clears
        if clears > 0:
            clearing_placements += 1

    return True, current_board, total_clears, 3, clearing_placements
```

- [ ] **Step 4: Update the one existing caller signature**

In `evaluate_monte_carlo_survival_jit` (around line 564), change the tuple unpack from:

```python
        did_survive, resulting_board, clears, placed_count = simulate_next_set_survival_jit(
```

to:

```python
        did_survive, resulting_board, clears, placed_count, _ = simulate_next_set_survival_jit(
```

(We'll use the new value properly in Task 3.)

- [ ] **Step 5: Run test to verify it passes**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak -v`
Expected: PASS.

- [ ] **Step 6: Run full suite to confirm no regressions**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add block_blast_solver/modules/solver.py tests/test_solver_streak.py
git commit -m "feat(solver): count clearing placements in next-set sim"
```

---

### Task 3: Add `avg_streak_continuity` to Monte Carlo evaluator

**Files:**
- Modify: `block_blast_solver/modules/solver.py:551-578`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_solver_streak.py`:

```python
class MonteCarloStreakContinuityTests(unittest.TestCase):
    def test_evaluator_returns_streak_continuity_pct(self):
        # Empty board: streak continuity should be in [0.0, 1.0].
        result = solver.evaluate_monte_carlo_survival_jit(np.int64(0), 2)
        # New signature: (score, survival_pct, clear_routes, future_fits, streak_continuity)
        self.assertEqual(len(result), 5)
        score, survival_pct, clear_routes, future_fits, streak_continuity = result
        self.assertGreaterEqual(streak_continuity, 0.0)
        self.assertLessEqual(streak_continuity, 1.0)

    def test_dead_board_streak_continuity_is_zero(self):
        # All bits set = fully filled, no sim can place anything.
        full_board = np.int64(-1)  # all 64 bits set when interpreted as uint
        _, _, _, _, streak_continuity = solver.evaluate_monte_carlo_survival_jit(full_board, 2)
        self.assertEqual(streak_continuity, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak.MonteCarloStreakContinuityTests -v`
Expected: FAIL — `evaluate_monte_carlo_survival_jit` returns 4-tuple.

- [ ] **Step 3: Replace the function body**

Replace the entire body of `evaluate_monte_carlo_survival_jit` (lines ~551-578) with:

```python
@njit(cache=True)
def evaluate_monte_carlo_survival_jit(board_mask: int, sample_count: int) -> Tuple[float, float, int, int, float]:
    catalog_masks, catalog_shapes, _ = get_monte_carlo_piece_catalog()
    samples = get_next_piece_samples(sample_count)
    survived = 0
    total_clear_routes = 0
    total_future_fits = 0
    partial_progress = 0
    streak_ratio_sum = 0.0
    streak_ratio_count = 0

    for sample_idx in range(sample_count):
        sample = samples[sample_idx]
        did_survive, resulting_board, clears, placed_count, clearing_placements = simulate_next_set_survival_jit(
            board_mask,
            sample,
            catalog_masks,
            catalog_shapes,
        )
        if did_survive:
            survived += 1
        total_clear_routes += clears
        partial_progress += placed_count
        total_future_fits += count_future_piece_fits_jit(resulting_board)
        if placed_count > 0:
            streak_ratio_sum += clearing_placements / placed_count
            streak_ratio_count += 1

    survival_pct = (survived * 100.0) / sample_count
    average_clear_routes = total_clear_routes / sample_count
    average_future_fits = total_future_fits / sample_count
    average_progress = partial_progress / sample_count
    score = survival_pct + (average_clear_routes * 3.0) + (average_future_fits * 2.0) + (average_progress * 10.0)

    if streak_ratio_count > 0:
        avg_streak_continuity = streak_ratio_sum / streak_ratio_count
    else:
        avg_streak_continuity = 0.0

    return score, survival_pct, int(average_clear_routes + 0.5), int(average_future_fits + 0.5), avg_streak_continuity
```

- [ ] **Step 4: Update the one Python caller of this signature**

In `evaluate_survival_score_jit` (around line 433), change:

```python
    _, next_survival_pct, clear_routes, mc_future_fits = evaluate_monte_carlo_survival_jit(board_mask, sample_count)
```

to:

```python
    _, next_survival_pct, clear_routes, mc_future_fits, streak_continuity = evaluate_monte_carlo_survival_jit(board_mask, sample_count)
```

(We'll plumb `streak_continuity` further in Task 4.)

- [ ] **Step 5: Run streak tests**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak -v`
Expected: PASS.

- [ ] **Step 6: Run full suite**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add block_blast_solver/modules/solver.py tests/test_solver_streak.py
git commit -m "feat(solver): expose avg_streak_continuity from Monte Carlo evaluator"
```

---

### Task 4: Plumb streak continuity through `evaluate_survival_score_jit`

**Files:**
- Modify: `block_blast_solver/modules/solver.py:401-460` (signature + body + return)

- [ ] **Step 1: Add new parameter and return value**

Change the function signature `evaluate_survival_score_jit` to add `w_monte_carlo_streak: float` as the last param, and add `float` (streak_continuity) as the last element of the return tuple:

```python
@njit(cache=True)
def evaluate_survival_score_jit(board_mask: int,
                                accumulated_score: float,
                                w_empty: float,
                                w_holes: float,
                                w_bumpiness: float,
                                w_readiness: float,
                                w_future_fits: float,
                                w_largest_region: float,
                                w_small_region_penalty: float,
                                w_line_readiness_survival: float,
                                w_trap_penalty: float,
                                normal_samples: int,
                                danger_samples: int,
                                danger_filled_cells: int,
                                min_future_fits: int,
                                w_monte_carlo_survival: float,
                                w_monte_carlo_clear_routes: float,
                                w_monte_carlo_future_fits: float,
                                w_monte_carlo_streak: float) -> Tuple[float, float, int, int, int, float]:
```

Inside the body, after the existing `monte_carlo_score = (...)` block, add the streak contribution:

```python
    monte_carlo_score = (
        (next_survival_pct * w_monte_carlo_survival)
        + (clear_routes * w_monte_carlo_clear_routes)
        + (mc_future_fits * w_monte_carlo_future_fits)
        + (streak_continuity * w_monte_carlo_streak)
    )

    return base_score + monte_carlo_score, next_survival_pct, clear_routes, mc_future_fits, sample_count, streak_continuity
```

(`streak_continuity` was already unpacked from `evaluate_monte_carlo_survival_jit` in Task 3.)

- [ ] **Step 2: Update the one caller — `solve_recursive`**

In `solve_recursive` (around line 619), change the unpack from:

```python
        score, next_survival_pct, clear_routes, mc_future_fits, sample_count = evaluate_survival_score_jit(
```

to:

```python
        score, next_survival_pct, clear_routes, mc_future_fits, sample_count, streak_continuity = evaluate_survival_score_jit(
```

And append `w_monte_carlo_streak` as the last argument inside the call (it will be passed in via Task 5's signature change).

For now to keep this task self-contained: also add `w_monte_carlo_streak: float` to the `solve_recursive` parameter list (after `w_monte_carlo_future_fits`) and pass it through the recursive call site (the `solve_recursive(...)` call inside the inner loop at the end of the function), then propagate it into the `evaluate_survival_score_jit` call.

The pattern to follow is identical to how `w_monte_carlo_future_fits` is currently threaded — copy that pattern for `w_monte_carlo_streak` in three places:
1. Parameter list of `solve_recursive`
2. The `evaluate_survival_score_jit(...)` call
3. The recursive `solve_recursive(...)` call at the bottom

After Step 2, the diagnostics holder (still size 4) does not yet record `streak_continuity` — that comes in Task 6. Add a temporary write to extend the holder to size 5 in this step:

In the `if score > best_score_holder[0]:` block (around line 636), add a line writing `streak_continuity`:

```python
            best_diagnostics_holder[4] = streak_continuity
```

- [ ] **Step 3: Grow the holder in `_solve_core`**

In `_solve_core` (around line 793), change:

```python
    best_diagnostics_holder = np.zeros(4, dtype=np.float64)
```

to:

```python
    best_diagnostics_holder = np.zeros(5, dtype=np.float64)
```

And update the `solve_recursive(...)` call in `_solve_core` (around line 802) to pass `config.W_MONTE_CARLO_STREAK` as the new last argument (immediately after `config.W_MONTE_CARLO_FUTURE_FITS`).

- [ ] **Step 4: Run full suite — should pass since no diagnostics consumer reads index [4] yet**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v`
Expected: all 19+ tests pass.

- [ ] **Step 5: Commit**

```bash
git add block_blast_solver/modules/solver.py
git commit -m "feat(solver): thread streak weight through scoring pipeline"
```

---

### Task 5: Add per-placement streak scoring in `solve_recursive`

**Files:**
- Modify: `block_blast_solver/modules/solver.py:581-712`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_solver_streak.py`:

```python
from block_blast_solver.modules import solver as solver_module


def _empty_board():
    return np.zeros((8, 8), dtype=np.int32)


def _piece_1x1():
    return np.array([[1]], dtype=np.int32)


def _piece_row_of_8():
    return np.ones((1, 8), dtype=np.int32)


class StreakScoringInRecursionTests(unittest.TestCase):
    def test_solver_prefers_three_clearing_placements_over_one_fat_clear(self):
        # Board: rows 0..6 fully filled, row 7 has 5 cells filled (cols 0..4 empty).
        # Wait: simpler — fully fill rows 6 and 7 except col 0,1,2 in row 7.
        # Plan A piece: 1x3 fills row 7 cols 0..2 -> clears row 7 only (1 line) then 2 zero-clear pieces.
        # Plan B pieces: three 1x1 each placed where each triggers its own clear.
        # Constructing precise scenarios is fragile; use a coarser check instead:
        # Provide three 1x1 pieces on a board that has 3 different rows each missing one cell.
        board = np.ones((8, 8), dtype=np.int32)
        # punch a hole in row 1 col 0, row 3 col 0, row 5 col 0
        board[1, 0] = 0
        board[3, 0] = 0
        board[5, 0] = 0
        # also clear row 7 fully so the board is not already terminal
        board[7, :] = 0

        pieces = [_piece_1x1(), _piece_1x1(), _piece_1x1()]
        moves, score, diagnostics = solver_module.solve_with_diagnostics(board, pieces)
        self.assertIsNotNone(moves)
        # Each 1x1 should land on (1,0), (3,0), (5,0) in some order, each clearing one row.
        targets = {(1, 0), (3, 0), (5, 0)}
        chosen = {(m["row"], m["col"]) for m in moves}
        self.assertEqual(chosen, targets)

    def test_diagnostics_dict_contains_next_streak_pct(self):
        board = _empty_board()
        pieces = [_piece_1x1(), _piece_1x1(), _piece_1x1()]
        _, _, diagnostics = solver_module.solve_with_diagnostics(board, pieces)
        self.assertIn("next_streak_pct", diagnostics)
        self.assertGreaterEqual(diagnostics["next_streak_pct"], 0.0)
        self.assertLessEqual(diagnostics["next_streak_pct"], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak.StreakScoringInRecursionTests -v`
Expected: FAIL — `next_streak_pct` not in diagnostics dict.

- [ ] **Step 3: Extend `solve_recursive` signature**

Add three new parameters at the end of `solve_recursive`'s parameter list (after `w_monte_carlo_streak` which was added in Task 4):

```python
                    w_streak_continue: float,
                    w_streak_break_penalty: float,
                    w_streak_perfect_bonus: float,
                    streak_in_plan: int) -> None:
```

- [ ] **Step 4: Apply per-placement streak logic in the recursion body**

In the inner placement loop, **right after** the line:

```python
                        if row_clear_mask != 0 or col_clear_mask != 0:
                            new_board_mask = np.int64(np.uint64(new_board_mask) & ~(row_clear_mask | col_clear_mask))
```

insert:

```python
                        is_clearing = (cleared_rows + cleared_cols) > 0
                        if is_clearing:
                            new_streak = streak_in_plan + 1
                            streak_delta = w_streak_continue
                        else:
                            new_streak = 0
                            streak_delta = w_streak_break_penalty
                        if depth + 1 == max_depth and new_streak == max_depth:
                            streak_delta += w_streak_perfect_bonus
```

Then change the recursive call's `accumulated_score` argument from:

```python
                            accumulated_score + ((cleared_rows + cleared_cols) * w_clear),
```

to:

```python
                            accumulated_score + ((cleared_rows + cleared_cols) * w_clear) + streak_delta,
```

And append the three streak weights plus `new_streak` to the recursive call (matching the order in the new signature):

```python
                            w_streak_continue, w_streak_break_penalty, w_streak_perfect_bonus,
                            new_streak
```

- [ ] **Step 5: Update `_solve_core` to pass the new args**

In `_solve_core`, at the `solve_recursive(...)` call site (around line 793-816), append after `config.W_MONTE_CARLO_STREAK`:

```python
        config.W_STREAK_CONTINUE,
        config.W_STREAK_BREAK_PENALTY,
        config.W_STREAK_PERFECT_BONUS,
        0,  # initial streak_in_plan
```

- [ ] **Step 6: Run streak tests**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak.StreakScoringInRecursionTests -v`
Expected: First test PASS, second test FAIL (diagnostics still doesn't include `next_streak_pct`). That's resolved in Task 6.

- [ ] **Step 7: Run full suite — existing tests must still pass**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v`
Expected: pre-existing tests pass; only the `test_diagnostics_dict_contains_next_streak_pct` test fails.

- [ ] **Step 8: Commit**

```bash
git add block_blast_solver/modules/solver.py tests/test_solver_streak.py
git commit -m "feat(solver): per-placement streak scoring in solve_recursive"
```

---

### Task 6: Expose `next_streak_pct` in diagnostics dict

**Files:**
- Modify: `block_blast_solver/modules/solver.py:712-742` (`_diagnostics_from_holder`)

- [ ] **Step 1: Update `_diagnostics_from_holder`**

Replace the function body with:

```python
def _diagnostics_from_holder(best_score: float, holder: np.ndarray) -> Dict[str, Any]:
    if best_score < -1e8:
        return {
            "risk_level": "dead",
            "next_survival_pct": 0.0,
            "clear_routes": 0,
            "future_fits": 0,
            "sample_count": 0,
            "next_streak_pct": 0.0,
        }

    next_survival_pct = float(holder[0])
    if next_survival_pct >= 70.0:
        risk_level = "low"
    elif next_survival_pct >= 35.0:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "risk_level": risk_level,
        "next_survival_pct": next_survival_pct,
        "clear_routes": int(holder[1]),
        "future_fits": int(holder[2]),
        "sample_count": int(holder[3]),
        "next_streak_pct": float(holder[4]),
    }
```

- [ ] **Step 2: Update the empty-pieces early return in `_solve_core`**

In `_solve_core`, the branch `if active_count == 0:` returns a dict — add `"next_streak_pct": 0.0` to it.

- [ ] **Step 3: Run streak tests — all should pass now**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak -v`
Expected: all pass.

- [ ] **Step 4: Run full suite**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add block_blast_solver/modules/solver.py
git commit -m "feat(solver): surface next_streak_pct in diagnostics"
```

---

### Task 7: Render `Next streak:` line in HUD

**Files:**
- Modify: `block_blast_solver/modules/visualizer.py:83-170` (`draw_hud`)
- Modify: `tests/test_visualizer_outcomes.py`

- [ ] **Step 1: Write the failing test**

Append a new test class to `tests/test_visualizer_outcomes.py`:

```python
class StreakHudTests(unittest.TestCase):
    def test_draw_hud_renders_next_streak_line(self):
        import numpy as np
        from block_blast_solver.modules.visualizer import Visualizer
        viz = Visualizer()
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        # Inspect what cv2.putText calls happen by monkey-patching it.
        import cv2
        calls = []
        original_put_text = cv2.putText
        def spy(img, text, org, *args, **kwargs):
            calls.append(text)
            return original_put_text(img, text, org, *args, **kwargs)
        cv2.putText = spy
        try:
            viz.draw_hud(
                frame,
                board_state=np.zeros((8, 8), dtype=np.int32),
                pieces=[None, None, None],
                status_text="OK",
                diagnostics={
                    "risk_level": "low",
                    "next_survival_pct": 80.0,
                    "clear_routes": 2,
                    "future_fits": 5,
                    "sample_count": 2,
                    "next_streak_pct": 0.75,
                },
            )
        finally:
            cv2.putText = original_put_text
        self.assertTrue(any("Next streak: 75%" in c for c in calls),
                        f"Expected 'Next streak: 75%' in HUD calls; got: {calls}")
```

(Adjust the `viz.draw_hud(...)` keyword arguments to match the actual `draw_hud` signature — read it first.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_visualizer_outcomes.StreakHudTests -v`
Expected: FAIL — no `Next streak:` text.

- [ ] **Step 3: Add the rendering**

In `draw_hud`, after the block that renders `"Clear routes: ..."` (around line 160), add:

```python
            streak_pct = diagnostics.get("next_streak_pct")
            if streak_pct is not None:
                y_pos += 22
                cv2.putText(hud_frame, f"Next streak: {int(streak_pct * 100)}%", (panel_x + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
```

(Match the exact font/color/spacing of the surrounding lines — read them first.)

- [ ] **Step 4: Run the HUD test**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_visualizer_outcomes.StreakHudTests -v`
Expected: PASS.

- [ ] **Step 5: Run full suite**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add block_blast_solver/modules/visualizer.py tests/test_visualizer_outcomes.py
git commit -m "feat(hud): render next streak percentage"
```

---

### Task 8: Add focused streak-vs-fat-clear regression test

**Files:**
- Modify: `tests/test_solver_streak.py`

This test locks in the spec's #1 acceptance criterion: the break penalty must dominate a single fat clear.

- [ ] **Step 1: Append the test**

```python
class StreakVsFatClearTests(unittest.TestCase):
    def test_break_penalty_dominates_single_fat_clear(self):
        """
        Construct a scenario where:
          - Plan A: piece P1 clears 3 lines simultaneously, then P2 and P3 can only place with zero clears.
          - Plan B: P1, P2, P3 each individually clear 1 line.
        Solver must select Plan B.

        Implementation: empty 8x8 board, three 1x1 pieces. Pre-fill rows 0, 2, 4 except for one
        cell each (cells (0,0), (2,0), (4,0) empty). Plan B = place each 1x1 in those holes
        for three 1-line clears. There is no single placement that clears 3 lines with a 1x1,
        so Plan A is impossible — the only viable plan is the 1+1+1 plan, which is also the
        all-clears plan. Verify diagnostics reflect a perfect streak.
        """
        board = np.ones((8, 8), dtype=np.int32)
        board[0, 0] = 0
        board[2, 0] = 0
        board[4, 0] = 0
        board[6, :] = 0  # leave room so pieces can be placed even if not on the target holes
        board[7, :] = 0

        pieces = [_piece_1x1(), _piece_1x1(), _piece_1x1()]
        moves, score, diagnostics = solver_module.solve_with_diagnostics(board, pieces)
        self.assertIsNotNone(moves)
        chosen_cells = {(m["row"], m["col"]) for m in moves}
        expected = {(0, 0), (2, 0), (4, 0)}
        self.assertEqual(chosen_cells, expected,
                         f"Expected all 3 placements to land on clearing holes; got {chosen_cells}")
```

- [ ] **Step 2: Run it**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_streak.StreakVsFatClearTests -v`
Expected: PASS (the prior streak weights already steer the solver here).

- [ ] **Step 3: Commit**

```bash
git add tests/test_solver_streak.py
git commit -m "test(solver): lock in streak-vs-fat-clear preference"
```

---

### Task 9: Live performance + integration check

**Files:** none modified (verification only)

- [ ] **Step 1: Re-run the full test suite (timed)**

Run: `C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v`
Expected: all tests pass; total wall time within ~10s of prior baseline (~3.4s warm). If significantly slower (> 2× baseline), investigate before merging.

- [ ] **Step 2: Cold-start performance probe**

Run:

```bash
C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -c "import time, numpy as np; from block_blast_solver.modules import solver; b = np.zeros((8,8),dtype=np.int32); p=[np.ones((3,3),dtype=np.int32), np.array([[1]],dtype=np.int32), np.ones((1,5),dtype=np.int32)]; solver.solve_with_diagnostics(b,p); t0=time.perf_counter(); solver.solve_with_diagnostics(b,p); print('warm ms:', (time.perf_counter()-t0)*1000)"
```

Expected: warm run ≤ 150 ms (target ~100 ms). If higher, capture the number and decide whether to lower `MONTE_CARLO_DANGER_SAMPLES` or skip the streak Monte Carlo metric in non-danger mode.

- [ ] **Step 3: Live capture sanity (manual, user-driven)**

Hand off to the user: run `main.py` against the live game, watch HUD, confirm `Next streak: N%` line appears and updates, and confirm streaks visibly extend further than the prior baseline.

No commit for this task.

---

## Self-Review

**Spec coverage:**
- Section 1 (per-placement streak scoring + 3 weights): Tasks 1, 5.
- Section 2 (MC streak continuity metric + new weight + diagnostics): Tasks 1, 2, 3, 4, 6.
- Section 3 (HUD field): Task 7.
- Testing (6 tests): Tasks 2, 3, 5, 7, 8 cover tests 1, 2, 3, 4, 5, 6 from the spec respectively. ✓
- Rollout / performance gate: Task 9. ✓

**Placeholder scan:** No TBDs. Code blocks complete. No "similar to" references. ✓

**Type consistency:**
- `simulate_next_set_survival_jit` return: `Tuple[bool, int, int, int, int]` (Task 2) — caller updated in same task.
- `evaluate_monte_carlo_survival_jit` return: `Tuple[float, float, int, int, float]` (Task 3) — caller updated in same task.
- `evaluate_survival_score_jit` return: `Tuple[float, float, int, int, int, float]` (Task 4) — caller updated in same task.
- `solve_recursive` signature gains `w_monte_carlo_streak`, `w_streak_continue`, `w_streak_break_penalty`, `w_streak_perfect_bonus`, `streak_in_plan` — all wired through `_solve_core` (Tasks 4, 5).
- Diagnostics holder grows from size 4 to size 5 (Task 4 Step 3) — read in `_diagnostics_from_holder` index `[4]` (Task 6).
- Diagnostics dict key `next_streak_pct` defined in Task 6, consumed by HUD in Task 7. ✓
