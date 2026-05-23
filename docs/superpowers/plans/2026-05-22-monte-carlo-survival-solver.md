# Monte Carlo Survival Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create git commits unless the user explicitly asks for commits.

**Goal:** Add bounded Monte Carlo next-piece simulation so the solver avoids late-game dead ends and prefers boards with high next-set survival probability.

**Architecture:** Keep the existing exact exhaustive search over the current visible pieces. At terminal depth, score each board with the existing survival evaluator plus deterministic next-set simulations from a curated piece catalog. Keep `solve(board, active_pieces) -> (moves, best_score)` unchanged and add a diagnostic path for HUD risk text.

**Tech Stack:** Python 3.13, NumPy, Numba, OpenCV HUD rendering, standard-library `unittest`.

---

## File Structure

- Modify: `block_blast_solver/config.py`
  - Add Monte Carlo sample counts, danger thresholds, and scoring weights.

- Modify: `block_blast_solver/modules/solver.py`
  - Add deterministic piece catalog helpers.
  - Add deterministic next-set samples.
  - Add next-set survival simulation helpers.
  - Add Monte Carlo terminal scoring.
  - Add diagnostics while preserving the existing `solve()` return shape.

- Modify: `block_blast_solver/modules/visualizer.py`
  - Accept optional solver diagnostics.
  - Display risk, next-set survival estimate, clear routes, and future fits in the side panel.

- Modify: `block_blast_solver/main.py`
  - Call the diagnostic solver wrapper.
  - Cache and pass diagnostics to the HUD.

- Create: `tests/test_solver_monte_carlo.py`
  - Deterministic unit tests for catalog, sample generation, Monte Carlo scoring, and danger behavior.

- Modify: `tests/test_visualizer_outcomes.py`
  - Add a HUD diagnostics smoke test.

---

### Task 1: Add Monte Carlo Solver Tests

**Files:**
- Create: `tests/test_solver_monte_carlo.py`

- [ ] **Step 1: Create the failing test file**

Create `tests/test_solver_monte_carlo.py` with this content:

```python
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER_ROOT = os.path.join(ROOT, "block_blast_solver")
if SOLVER_ROOT not in sys.path:
    sys.path.insert(0, SOLVER_ROOT)

from modules import solver


def empty_board():
    return np.zeros((8, 8), dtype=np.uint8)


def piece(matrix):
    return np.array(matrix, dtype=np.uint8)


def board_mask(board):
    mask = 0
    for row in range(8):
        for col in range(8):
            if board[row, col] == 1:
                mask |= 1 << (row * 8 + col)
    return mask


class MonteCarloSolverTests(unittest.TestCase):
    def test_piece_catalog_contains_rotated_and_mirrored_families(self):
        masks, shapes, families = solver.get_monte_carlo_piece_catalog()

        self.assertGreaterEqual(masks.shape[0], 24)
        self.assertEqual(masks.shape[0], shapes.shape[0])
        self.assertEqual(masks.shape[0], families.shape[0])
        self.assertIn(9, set(int(family) for family in families))
        self.assertIn(25, set(int(family) for family in families))

    def test_next_piece_samples_are_deterministic(self):
        first = solver.get_next_piece_samples(24)
        second = solver.get_next_piece_samples(24)

        self.assertEqual(first.shape, (24, 3))
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(int(np.max(first)), 0)

    def test_monte_carlo_rewards_more_playable_next_sets(self):
        open_board = empty_board()
        trapped_board = np.ones((8, 8), dtype=np.uint8)
        trapped_board[0, 0] = 0
        trapped_board[0, 1] = 0
        trapped_board[1, 0] = 0
        trapped_board[7, 7] = 0

        open_score, open_survival, open_routes, open_fits = solver.evaluate_monte_carlo_survival_jit(board_mask(open_board), 24)
        trapped_score, trapped_survival, trapped_routes, trapped_fits = solver.evaluate_monte_carlo_survival_jit(board_mask(trapped_board), 24)

        self.assertGreater(open_survival, trapped_survival)
        self.assertGreater(open_fits, trapped_fits)
        self.assertGreater(open_score, trapped_score)
        self.assertGreaterEqual(open_routes, trapped_routes)

    def test_solver_prefers_high_survival_board_over_cosmetic_immediate_clear(self):
        board = empty_board()
        board[7, 0:7] = 1
        board[0:5, 0:5] = 1
        board[0:3, 5:8] = 0
        board[5:8, 5:8] = 0

        pieces = [
            piece([[1]]),
            piece([[1, 1], [1, 0]]),
            piece([[1, 1], [1, 0]]),
        ]

        moves, score, diagnostics = solver.solve_with_diagnostics(board, pieces)

        self.assertIsNotNone(moves)
        self.assertGreater(score, -1e8)
        self.assertGreaterEqual(diagnostics["next_survival_pct"], 20.0)
        self.assertGreaterEqual(diagnostics["future_fits"], 1)

    def test_diagnostics_are_zero_when_no_legal_move_exists(self):
        board = np.ones((8, 8), dtype=np.uint8)
        pieces = [piece([[1]]), None, None]

        moves, score, diagnostics = solver.solve_with_diagnostics(board, pieces)

        self.assertIsNone(moves)
        self.assertLess(score, -1e8)
        self.assertEqual(diagnostics["risk_level"], "dead")
        self.assertEqual(diagnostics["next_survival_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_monte_carlo -v
```

Expected: tests fail because `get_monte_carlo_piece_catalog`, `get_next_piece_samples`, `evaluate_monte_carlo_survival_jit`, and `solve_with_diagnostics` do not exist yet.

---

### Task 2: Add Monte Carlo Config Values

**Files:**
- Modify: `block_blast_solver/config.py`

- [ ] **Step 1: Add config constants after the existing survival weights**

Add this block after `W_TRAP_PENALTY = -55.0`:

```python
# Monte Carlo next-piece survival weights and limits
MONTE_CARLO_NORMAL_SAMPLES = 2
MONTE_CARLO_DANGER_SAMPLES = 4
MONTE_CARLO_DANGER_FILLED_CELLS = 38
MONTE_CARLO_MIN_FUTURE_FITS = 8
W_MONTE_CARLO_SURVIVAL = 120.0
W_MONTE_CARLO_CLEAR_ROUTES = 18.0
W_MONTE_CARLO_FUTURE_FITS = 12.0
```

- [ ] **Step 2: Compile-check config**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/config.py
```

Expected: command exits with code 0.

---

### Task 3: Add Piece Catalog and Deterministic Samples

**Files:**
- Modify: `block_blast_solver/modules/solver.py`

- [ ] **Step 1: Add catalog helper above `count_future_piece_fits_jit`**

Add this code above the existing `get_future_piece_masks()` function:

```python
@njit(cache=True)
def get_monte_carlo_piece_catalog() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks = np.zeros(28, dtype=np.uint64)
    shapes = np.zeros((28, 2), dtype=np.int32)
    families = np.zeros(28, dtype=np.int32)

    masks[0] = np.uint64(0x1); shapes[0, 0] = 1; shapes[0, 1] = 1; families[0] = 1
    masks[1] = np.uint64(0x3); shapes[1, 0] = 1; shapes[1, 1] = 2; families[1] = 2
    masks[2] = np.uint64(0x101); shapes[2, 0] = 2; shapes[2, 1] = 1; families[2] = 2
    masks[3] = np.uint64(0x7); shapes[3, 0] = 1; shapes[3, 1] = 3; families[3] = 3
    masks[4] = np.uint64(0x10101); shapes[4, 0] = 3; shapes[4, 1] = 1; families[4] = 3
    masks[5] = np.uint64(0xF); shapes[5, 0] = 1; shapes[5, 1] = 4; families[5] = 4
    masks[6] = np.uint64(0x1010101); shapes[6, 0] = 4; shapes[6, 1] = 1; families[6] = 4
    masks[7] = np.uint64(0x1F); shapes[7, 0] = 1; shapes[7, 1] = 5; families[7] = 5
    masks[8] = np.uint64(0x101010101); shapes[8, 0] = 5; shapes[8, 1] = 1; families[8] = 5
    masks[9] = np.uint64(0x303); shapes[9, 0] = 2; shapes[9, 1] = 2; families[9] = 9
    masks[10] = np.uint64(0x70707); shapes[10, 0] = 3; shapes[10, 1] = 3; families[10] = 25

    masks[11] = np.uint64(0x107); shapes[11, 0] = 2; shapes[11, 1] = 3; families[11] = 11
    masks[12] = np.uint64(0x701); shapes[12, 0] = 2; shapes[12, 1] = 3; families[12] = 11
    masks[13] = np.uint64(0x30102); shapes[13, 0] = 3; shapes[13, 1] = 2; families[13] = 11
    masks[14] = np.uint64(0x20103); shapes[14, 0] = 3; shapes[14, 1] = 2; families[14] = 11
    masks[15] = np.uint64(0x407); shapes[15, 0] = 2; shapes[15, 1] = 3; families[15] = 11
    masks[16] = np.uint64(0x704); shapes[16, 0] = 2; shapes[16, 1] = 3; families[16] = 11
    masks[17] = np.uint64(0x10203); shapes[17, 0] = 3; shapes[17, 1] = 2; families[17] = 11
    masks[18] = np.uint64(0x30201); shapes[18, 0] = 3; shapes[18, 1] = 2; families[18] = 11

    masks[19] = np.uint64(0x207); shapes[19, 0] = 2; shapes[19, 1] = 3; families[19] = 19
    masks[20] = np.uint64(0x702); shapes[20, 0] = 2; shapes[20, 1] = 3; families[20] = 19
    masks[21] = np.uint64(0x10203); shapes[21, 0] = 3; shapes[21, 1] = 2; families[21] = 19
    masks[22] = np.uint64(0x30102); shapes[22, 0] = 3; shapes[22, 1] = 2; families[22] = 19

    masks[23] = np.uint64(0x306); shapes[23, 0] = 2; shapes[23, 1] = 3; families[23] = 23
    masks[24] = np.uint64(0x603); shapes[24, 0] = 2; shapes[24, 1] = 3; families[24] = 23
    masks[25] = np.uint64(0x10206); shapes[25, 0] = 3; shapes[25, 1] = 2; families[25] = 23
    masks[26] = np.uint64(0x20103); shapes[26, 0] = 3; shapes[26, 1] = 2; families[26] = 23

    masks[27] = np.uint64(0x707); shapes[27, 0] = 2; shapes[27, 1] = 3; families[27] = 27
    return masks, shapes, families
```

- [ ] **Step 2: Add deterministic sample helper below the catalog**

Add:

```python
@njit(cache=True)
def get_next_piece_samples(sample_count: int) -> np.ndarray:
    masks, _, _ = get_monte_carlo_piece_catalog()
    catalog_size = masks.shape[0]
    samples = np.zeros((sample_count, 3), dtype=np.int32)

    for sample_idx in range(sample_count):
        samples[sample_idx, 0] = (sample_idx * 7 + 3) % catalog_size
        samples[sample_idx, 1] = (sample_idx * 11 + 5) % catalog_size
        samples[sample_idx, 2] = (sample_idx * 17 + 9) % catalog_size

    return samples
```

- [ ] **Step 3: Run the catalog-focused tests**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_monte_carlo.MonteCarloSolverTests.test_piece_catalog_contains_rotated_and_mirrored_families tests.test_solver_monte_carlo.MonteCarloSolverTests.test_next_piece_samples_are_deterministic -v
```

Expected: these two tests pass.

---

### Task 4: Add Next-Set Survival Simulation

**Files:**
- Modify: `block_blast_solver/modules/solver.py`

- [ ] **Step 1: Add clear helper above `solve_recursive`**

Add:

```python
@njit(cache=True)
def clear_completed_lines_jit(board_mask: int, row_start: int, row_count: int, col_start: int, col_count: int) -> Tuple[int, int]:
    u_board = np.uint64(board_mask)
    row_clear_mask = np.uint64(0)
    col_clear_mask = np.uint64(0)
    clear_count = 0

    for row_offset in range(row_count):
        row_idx = row_start + row_offset
        if row_idx < 0 or row_idx >= 8:
            continue
        mask = np.uint64(0xFF) << np.uint64(row_idx * 8)
        if (u_board & mask) == mask:
            row_clear_mask |= mask
            clear_count += 1

    for col_offset in range(col_count):
        col_idx = col_start + col_offset
        if col_idx < 0 or col_idx >= 8:
            continue
        mask = np.uint64(0x0101010101010101) << np.uint64(col_idx)
        if (u_board & mask) == mask:
            col_clear_mask |= mask
            clear_count += 1

    if row_clear_mask != 0 or col_clear_mask != 0:
        u_board = u_board & ~(row_clear_mask | col_clear_mask)

    return int(np.int64(u_board)), clear_count
```

- [ ] **Step 2: Add single-piece best placement simulator**

Add below the clear helper:

```python
@njit(cache=True)
def simulate_best_single_future_piece_jit(board_mask: int, piece_mask: int, ph: int, pw: int) -> Tuple[bool, int, int]:
    u_board = np.uint64(board_mask)
    best_board = np.int64(0)
    best_score = -1000000000
    best_clears = 0
    found = False

    for row in range(9 - ph):
        for col in range(9 - pw):
            shifted_piece = np.uint64(piece_mask) << np.uint64(row * 8 + col)
            if (u_board & shifted_piece) != np.uint64(0):
                continue

            placed_board = np.int64(board_mask) | np.int64(shifted_piece)
            cleared_board, clear_count = clear_completed_lines_jit(placed_board, row, ph, col, pw)
            empty_cells = 64 - popcount(cleared_board)
            future_fits = count_future_piece_fits_jit(cleared_board)
            largest_region, small_regions, region_count = calculate_empty_regions_jit(cleared_board)
            score = (clear_count * 1000) + (future_fits * 18) + (largest_region * 8) + (empty_cells * 5) - (small_regions * 80) - (max(0, region_count - 1) * 25)

            if score > best_score:
                best_score = score
                best_board = cleared_board
                best_clears = clear_count
                found = True

    return found, int(best_board), best_clears
```

- [ ] **Step 3: Add next-set survival simulator**

Add below the single-piece simulator:

```python
@njit(cache=True)
def simulate_next_set_survival_jit(board_mask: int, sample: np.ndarray, catalog_masks: np.ndarray, catalog_shapes: np.ndarray) -> Tuple[bool, int, int, int]:
    current_board = board_mask
    total_clears = 0

    for sample_pos in range(3):
        piece_idx = sample[sample_pos]
        piece_mask = int(catalog_masks[piece_idx])
        ph = int(catalog_shapes[piece_idx, 0])
        pw = int(catalog_shapes[piece_idx, 1])
        placed, next_board, clears = simulate_best_single_future_piece_jit(current_board, piece_mask, ph, pw)
        if not placed:
            return False, current_board, total_clears, sample_pos
        current_board = next_board
        total_clears += clears

    return True, current_board, total_clears, 3
```

- [ ] **Step 4: Add Monte Carlo evaluator**

Add below `simulate_next_set_survival_jit`:

```python
@njit(cache=True)
def evaluate_monte_carlo_survival_jit(board_mask: int, sample_count: int) -> Tuple[float, float, int, int]:
    catalog_masks, catalog_shapes, _ = get_monte_carlo_piece_catalog()
    samples = get_next_piece_samples(sample_count)
    survived = 0
    total_clear_routes = 0
    total_future_fits = 0
    partial_progress = 0

    for sample_idx in range(sample_count):
        sample = samples[sample_idx]
        did_survive, resulting_board, clears, placed_count = simulate_next_set_survival_jit(board_mask, sample, catalog_masks, catalog_shapes)
        if did_survive:
            survived += 1
        total_clear_routes += clears
        partial_progress += placed_count
        total_future_fits += count_future_piece_fits_jit(resulting_board)

    survival_pct = (survived * 100.0) / sample_count
    average_clear_routes = total_clear_routes / sample_count
    average_future_fits = total_future_fits / sample_count
    average_progress = partial_progress / sample_count
    score = survival_pct + (average_clear_routes * 3.0) + (average_future_fits * 2.0) + (average_progress * 10.0)

    return score, survival_pct, int(round(average_clear_routes)), int(round(average_future_fits))
```

- [ ] **Step 5: Run the Monte Carlo evaluator tests**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_monte_carlo.MonteCarloSolverTests.test_monte_carlo_rewards_more_playable_next_sets -v
```

Expected: test passes.

---

### Task 5: Wire Monte Carlo Into Terminal Scoring and Diagnostics

**Files:**
- Modify: `block_blast_solver/modules/solver.py`

- [ ] **Step 1: Add sample count selector above `evaluate_survival_score_jit`**

Add:

```python
@njit(cache=True)
def choose_monte_carlo_sample_count_jit(board_mask: int,
                                        normal_samples: int,
                                        danger_samples: int,
                                        danger_filled_cells: int,
                                        min_future_fits: int) -> int:
    filled_cells = popcount(board_mask)
    future_fits = count_future_piece_fits_jit(board_mask)
    readiness = check_readiness_jit(board_mask, 64 - filled_cells)
    _, _, region_count = calculate_empty_regions_jit(board_mask)

    if filled_cells >= danger_filled_cells:
        return danger_samples
    if future_fits <= min_future_fits:
        return danger_samples
    if readiness < 1.0:
        return danger_samples
    if region_count >= 4:
        return danger_samples
    return normal_samples
```

- [ ] **Step 2: Extend `evaluate_survival_score_jit` signature and body**

Change the signature to include Monte Carlo config values:

```python
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
                                w_monte_carlo_future_fits: float) -> Tuple[float, float, int, int, int]:
```

Replace its return block with:

```python
    sample_count = choose_monte_carlo_sample_count_jit(
        board_mask,
        normal_samples,
        danger_samples,
        danger_filled_cells,
        min_future_fits,
    )
    _, next_survival_pct, clear_routes, mc_future_fits = evaluate_monte_carlo_survival_jit(board_mask, sample_count)

    base_score = (
        (accumulated_score * 0.75)
        + (w_empty * empty_cells)
        + (w_holes * holes)
        + (w_bumpiness * bumpiness)
        + (w_readiness * readiness)
        + (w_future_fits * future_fits)
        + (w_largest_region * largest_region)
        + (w_small_region_penalty * small_regions)
        + (w_line_readiness_survival * line_readiness)
        + (w_trap_penalty * traps)
        - fragmentation_penalty
    )
    monte_carlo_score = (
        (next_survival_pct * w_monte_carlo_survival)
        + (clear_routes * w_monte_carlo_clear_routes)
        + (mc_future_fits * w_monte_carlo_future_fits)
    )

    return base_score + monte_carlo_score, next_survival_pct, clear_routes, mc_future_fits, sample_count
```

- [ ] **Step 3: Add diagnostics holder to `solve_recursive`**

Add a new parameter after `best_score_holder`:

```python
best_diagnostics_holder: np.ndarray,
```

Add config parameters at the end of the recursive signature:

```python
normal_samples: int,
danger_samples: int,
danger_filled_cells: int,
min_future_fits: int,
w_monte_carlo_survival: float,
w_monte_carlo_clear_routes: float,
w_monte_carlo_future_fits: float
```

At terminal depth, replace:

```python
        score = evaluate_survival_score_jit(
```

with:

```python
        score, next_survival_pct, clear_routes, mc_future_fits, sample_count = evaluate_survival_score_jit(
```

When a new best score is found, set diagnostics:

```python
            best_diagnostics_holder[0] = next_survival_pct
            best_diagnostics_holder[1] = clear_routes
            best_diagnostics_holder[2] = mc_future_fits
            best_diagnostics_holder[3] = sample_count
```

Pass the new parameters through the recursive call.

- [ ] **Step 4: Add Python diagnostics conversion helper below `solve_recursive`**

Add:

```python
def _diagnostics_from_holder(best_score: float, holder: np.ndarray) -> Dict[str, Any]:
    if best_score < -1e8:
        return {
            "risk_level": "dead",
            "next_survival_pct": 0.0,
            "clear_routes": 0,
            "future_fits": 0,
            "sample_count": 0,
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
    }
```

- [ ] **Step 5: Refactor `solve` through `_solve_core` and add `solve_with_diagnostics`**

Create a private helper with the old `solve` body and return diagnostics:

```python
def _solve_core(board: np.ndarray, active_pieces: List[Optional[np.ndarray]]) -> Tuple[Optional[List[Dict[str, Any]]], float, Dict[str, Any]]:
```

Inside it, create:

```python
    best_diagnostics_holder = np.zeros(4, dtype=np.float64)
```

Pass `best_diagnostics_holder` and all Monte Carlo config values to `solve_recursive`.

At the end, return:

```python
    diagnostics = _diagnostics_from_holder(best_score, best_diagnostics_holder)
    return moves, best_score, diagnostics
```

Replace public `solve` with:

```python
def solve(board: np.ndarray, active_pieces: List[Optional[np.ndarray]]) -> Tuple[Optional[List[Dict[str, Any]]], float]:
    moves, best_score, _ = _solve_core(board, active_pieces)
    return moves, best_score


def solve_with_diagnostics(board: np.ndarray, active_pieces: List[Optional[np.ndarray]]) -> Tuple[Optional[List[Dict[str, Any]]], float, Dict[str, Any]]:
    return _solve_core(board, active_pieces)
```

- [ ] **Step 6: Run solver Monte Carlo tests**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_monte_carlo -v
```

Expected: all tests in `tests.test_solver_monte_carlo` pass.

---

### Task 6: Add HUD Diagnostics Display

**Files:**
- Modify: `block_blast_solver/modules/visualizer.py`
- Modify: `tests/test_visualizer_outcomes.py`

- [ ] **Step 1: Add a visualizer test for diagnostics rendering**

Append this test to `VisualizerOutcomeTests` in `tests/test_visualizer_outcomes.py`:

```python
    def test_draw_hud_accepts_solver_diagnostics(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        board = np.zeros((8, 8), dtype=np.uint8)
        pieces = [None, None, None]
        diagnostics = {
            "risk_level": "high",
            "next_survival_pct": 28.5,
            "clear_routes": 2,
            "future_fits": 9,
            "sample_count": 64,
        }

        hud = Visualizer().draw_hud(frame, board, pieces, None, 0.0, False, diagnostics)

        self.assertEqual(hud.shape[0], frame.shape[0])
        self.assertEqual(hud.shape[1], frame.shape[1] + 320)
```

- [ ] **Step 2: Run the visualizer test and verify it fails**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_visualizer_outcomes.VisualizerOutcomeTests.test_draw_hud_accepts_solver_diagnostics -v
```

Expected: test fails because `draw_hud` does not accept the diagnostics parameter yet.

- [ ] **Step 3: Extend `draw_hud` signature**

Change the signature to:

```python
    def draw_hud(self,
                 frame: np.ndarray,
                 board_state: np.ndarray,
                 pieces: List[Optional[np.ndarray]],
                 moves: Optional[List[Dict[str, Any]]],
                 score: float,
                 occluded: bool,
                 diagnostics: Optional[Dict[str, Any]] = None) -> np.ndarray:
```

- [ ] **Step 4: Render diagnostics under the status line**

After the status text is drawn, add:

```python
        if diagnostics is not None:
            risk_level = str(diagnostics.get("risk_level", "unknown"))
            survival_pct = float(diagnostics.get("next_survival_pct", 0.0))
            clear_routes = int(diagnostics.get("clear_routes", 0))
            future_fits = int(diagnostics.get("future_fits", 0))
            sample_count = int(diagnostics.get("sample_count", 0))
            risk_color = (0, 255, 0)
            if risk_level == "medium":
                risk_color = (0, 200, 255)
            elif risk_level in ("high", "dead"):
                risk_color = (0, 0, 255)

            y_pos += 22
            cv2.putText(hud_frame, f"Risk: {risk_level.upper()}", (panel_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, risk_color, 1, cv2.LINE_AA)
            y_pos += 20
            cv2.putText(hud_frame, f"Next survival: {survival_pct:.0f}% ({sample_count})", (panel_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            y_pos += 18
            cv2.putText(hud_frame, f"Clear routes: {clear_routes} | Fits: {future_fits}", (panel_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
```

- [ ] **Step 5: Run visualizer tests**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_visualizer_outcomes -v
```

Expected: all visualizer tests pass.

---

### Task 7: Wire Diagnostics Through Main Loop

**Files:**
- Modify: `block_blast_solver/main.py`

- [ ] **Step 1: Add `last_diagnostics` cache near `last_score`**

After:

```python
    last_score = 0.0
```

Add:

```python
    last_diagnostics = None
```

- [ ] **Step 2: Use diagnostic solver wrapper**

Replace:

```python
                    moves, score = solver.solve(board_state, pieces)
```

with:

```python
                    moves, score, diagnostics = solver.solve_with_diagnostics(board_state, pieces)
```

After `last_score = score`, add:

```python
                    last_diagnostics = diagnostics
```

- [ ] **Step 3: Reset diagnostics when no pieces or recalibrating**

Where `last_moves = None` and `last_score = 0.0` are reset, also set:

```python
                    last_diagnostics = None
```

For each calibration reset block, add:

```python
                    last_diagnostics = None
```

or:

```python
            last_diagnostics = None
```

matching the indentation of the surrounding reset code.

- [ ] **Step 4: Pass diagnostics to HUD**

Replace each call shaped like:

```python
visualizer.draw_hud(frame, board_state, pieces, last_moves, last_score, occluded)
```

with:

```python
visualizer.draw_hud(frame, board_state, pieces, last_moves, last_score, occluded, last_diagnostics)
```

- [ ] **Step 5: Compile-check main**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/main.py
```

Expected: command exits with code 0.

---

### Task 8: Full Verification and Runtime Smoke

**Files:**
- Verify only; no planned edits.

- [ ] **Step 1: Run all unit tests**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Compile all changed Python files**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/main.py block_blast_solver/config.py block_blast_solver/modules/solver.py block_blast_solver/modules/vision.py block_blast_solver/modules/visualizer.py block_blast_solver/modules/capture.py tests/test_solver_survival.py tests/test_solver_monte_carlo.py tests/test_visualizer_outcomes.py tests/test_vision_pieces.py
```

Expected: command exits with code 0 and prints no errors.

- [ ] **Step 3: Run warm solver runtime smoke**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -c "import os, sys, time, numpy as np; sys.path.insert(0, os.path.abspath('block_blast_solver')); from modules import solver; board=np.zeros((8,8), dtype=np.uint8); board[0:5,0:4]=1; board[6,0:6]=1; pieces=[np.ones((1,5), dtype=np.uint8), np.ones((2,3), dtype=np.uint8), np.array([[1,0],[1,1]], dtype=np.uint8)]; solver.solve_with_diagnostics(board,pieces); start=time.time(); moves, score, diagnostics=solver.solve_with_diagnostics(board,pieces); print(moves); print(score); print(diagnostics); print(f'{(time.time()-start)*1000:.1f}ms')"
```

Expected: command prints moves or `None`, a score, diagnostics containing `next_survival_pct`, and warm runtime under 250 ms on this synthetic board.

- [ ] **Step 4: Check VS Code diagnostics**

Use the editor diagnostics tool for:

- `block_blast_solver/modules/solver.py`
- `block_blast_solver/modules/visualizer.py`
- `block_blast_solver/main.py`
- `tests/test_solver_monte_carlo.py`

Expected: no errors.

- [ ] **Step 5: Report status without committing**

Run:

```powershell
git status --short
```

Expected: changed files are visible. Do not commit unless the user explicitly asks.
