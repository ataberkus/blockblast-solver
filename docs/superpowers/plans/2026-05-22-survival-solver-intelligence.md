# Survival Solver Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create git commits unless the user explicitly asks for commits.

**Goal:** Upgrade the Block Blast solver so it prefers long-term survival and future piece flexibility over short-term greedy placements.

**Architecture:** Keep the existing bitmask + Numba exhaustive search over the current visible pieces. Replace the final board evaluator with a richer survival evaluator that scores future-piece fit coverage, connected empty space, line readiness, and traps. Add deterministic standard-library tests that call `solver.solve()` directly without camera input.

**Tech Stack:** Python 3.13, NumPy, Numba, standard-library `unittest`.

---

## File Structure

- Modify: `block_blast_solver/config.py`
  - Add survival scoring weights.
  - Keep existing weights for compatibility.

- Modify: `block_blast_solver/modules/solver.py`
  - Add JIT helper functions for future-piece fit count, empty-region scoring, line readiness, and trap penalties.
  - Replace terminal scoring in `solve_recursive()` with a new survival evaluator.
  - Keep `solve()` public API unchanged.

- Create: `tests/test_solver_survival.py`
  - Deterministic tests for solver behavior using direct board/piece matrices.
  - Uses `unittest`, so no new dependency is required.

---

### Task 1: Add Solver Regression Tests

**Files:**
- Create: `tests/test_solver_survival.py`

- [ ] **Step 1: Create the test directory**

Run:

```powershell
New-Item -ItemType Directory -Force -Path tests | Out-Null
```

Expected: command exits with code 0.

- [ ] **Step 2: Write failing tests for survival behavior**

Create `tests/test_solver_survival.py` with:

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


class SurvivalSolverTests(unittest.TestCase):
    def test_never_places_on_occupied_cells(self):
        board = empty_board()
        board[0, 0] = 1
        board[0, 1] = 1
        board[1, 0] = 1

        pieces = [piece([[1, 1], [1, 1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNotNone(moves)
        move = moves[0]
        row = move["row"]
        col = move["col"]
        placed = pieces[move["slot_index"]]

        for r in range(placed.shape[0]):
            for c in range(placed.shape[1]):
                if placed[r, c] == 1:
                    self.assertEqual(board[row + r, col + c], 0)
        self.assertGreater(score, -1e8)

    def test_preserves_open_three_by_three_space(self):
        board = np.ones((8, 8), dtype=np.uint8)
        board[0:3, 0:3] = 0
        board[5:8, 5:8] = 0
        board[6, 6] = 1

        pieces = [piece([[1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNotNone(moves)
        move = moves[0]
        self.assertFalse(0 <= move["row"] <= 2 and 0 <= move["col"] <= 2)
        self.assertGreater(score, -1e8)

    def test_prefers_clear_when_board_is_constrained(self):
        board = empty_board()
        board[0, 0:7] = 1
        board[1:8, :] = 1
        board[1, 0] = 0
        board[2, 0] = 0

        pieces = [piece([[1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNotNone(moves)
        self.assertEqual((moves[0]["row"], moves[0]["col"]), (0, 7))
        self.assertGreater(score, -1e8)

    def test_returns_none_when_no_legal_move_exists(self):
        board = np.ones((8, 8), dtype=np.uint8)
        pieces = [piece([[1]]), None, None]
        moves, score = solver.solve(board, pieces)

        self.assertIsNone(moves)
        self.assertLess(score, -1e8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests and verify current solver behavior**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_survival -v
```

Expected: at least one survival-specific test fails before the new evaluator is implemented. The occupied-cell and no-legal-move tests should pass.

---

### Task 2: Add Survival Scoring Weights

**Files:**
- Modify: `block_blast_solver/config.py`

- [ ] **Step 1: Add new weights after existing heuristic weights**

In `block_blast_solver/config.py`, after `W_READINESS = 350.0`, add:

```python
# Survival-focused evaluator weights
W_FUTURE_FITS = 42.0
W_LARGEST_REGION = 18.0
W_SMALL_REGION_PENALTY = -35.0
W_LINE_READINESS_SURVIVAL = 28.0
W_TRAP_PENALTY = -55.0
```

- [ ] **Step 2: Compile-check config**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/config.py
```

Expected: command exits with code 0 and prints no errors.

---

### Task 3: Add Future Piece Fit Scoring

**Files:**
- Modify: `block_blast_solver/modules/solver.py`

- [ ] **Step 1: Add future piece mask constants helper**

In `solver.py`, above `check_readiness_jit`, add:

```python
@njit(cache=True)
def get_future_piece_masks() -> Tuple[np.ndarray, np.ndarray]:
    masks = np.zeros(16, dtype=np.uint64)
    shapes = np.zeros((16, 2), dtype=np.int32)

    masks[0] = np.uint64(0x1); shapes[0, 0] = 1; shapes[0, 1] = 1
    masks[1] = np.uint64(0x3); shapes[1, 0] = 1; shapes[1, 1] = 2
    masks[2] = np.uint64(0x101); shapes[2, 0] = 2; shapes[2, 1] = 1
    masks[3] = np.uint64(0x7); shapes[3, 0] = 1; shapes[3, 1] = 3
    masks[4] = np.uint64(0x10101); shapes[4, 0] = 3; shapes[4, 1] = 1
    masks[5] = np.uint64(0xF); shapes[5, 0] = 1; shapes[5, 1] = 4
    masks[6] = np.uint64(0x1010101); shapes[6, 0] = 4; shapes[6, 1] = 1
    masks[7] = np.uint64(0x1F); shapes[7, 0] = 1; shapes[7, 1] = 5
    masks[8] = np.uint64(0x101010101); shapes[8, 0] = 5; shapes[8, 1] = 1
    masks[9] = np.uint64(0x303); shapes[9, 0] = 2; shapes[9, 1] = 2
    masks[10] = np.uint64(0x70707); shapes[10, 0] = 3; shapes[10, 1] = 3
    masks[11] = np.uint64(0x107); shapes[11, 0] = 2; shapes[11, 1] = 3
    masks[12] = np.uint64(0x701); shapes[12, 0] = 2; shapes[12, 1] = 3
    masks[13] = np.uint64(0x30102); shapes[13, 0] = 3; shapes[13, 1] = 2
    masks[14] = np.uint64(0x20103); shapes[14, 0] = 3; shapes[14, 1] = 2
    masks[15] = np.uint64(0x702); shapes[15, 0] = 2; shapes[15, 1] = 3

    return masks, shapes
```

- [ ] **Step 2: Add future fit counter**

Below the helper, add:

```python
@njit(cache=True)
def count_future_piece_fits_jit(board_mask: int) -> int:
    future_masks, future_shapes = get_future_piece_masks()
    u_board = np.uint64(board_mask)
    fit_score = 0

    for i in range(future_masks.shape[0]):
        piece_mask = future_masks[i]
        ph = future_shapes[i, 0]
        pw = future_shapes[i, 1]
        placements = 0

        for r in range(9 - ph):
            for c in range(9 - pw):
                shifted_piece = piece_mask << np.uint64(r * 8 + c)
                if (u_board & shifted_piece) == np.uint64(0):
                    placements += 1

        if placements > 0:
            fit_score += 8
            if placements > 2:
                fit_score += 4
            if placements > 6:
                fit_score += 3

    return fit_score
```

- [ ] **Step 3: Compile-check solver**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/modules/solver.py
```

Expected: command exits with code 0.

---

### Task 4: Add Empty Region and Trap Scoring

**Files:**
- Modify: `block_blast_solver/modules/solver.py`

- [ ] **Step 1: Add connected-region scoring**

Below `calculate_bumpiness_jit`, add:

```python
@njit(cache=True)
def calculate_empty_regions_jit(board_mask: int) -> Tuple[int, int, int]:
    u_board = np.uint64(board_mask)
    visited = np.uint64(0)
    largest_region = 0
    small_regions = 0
    region_count = 0

    for start in range(64):
        start_bit = np.uint64(1) << np.uint64(start)
        if (u_board & start_bit) != np.uint64(0) or (visited & start_bit) != np.uint64(0):
            continue

        region_count += 1
        stack = np.zeros(64, dtype=np.int32)
        stack_size = 1
        stack[0] = start
        visited |= start_bit
        region_size = 0

        while stack_size > 0:
            stack_size -= 1
            idx = stack[stack_size]
            region_size += 1
            row = idx // 8
            col = idx % 8

            neighbors = np.zeros(4, dtype=np.int32)
            neighbor_count = 0
            if row > 0:
                neighbors[neighbor_count] = idx - 8
                neighbor_count += 1
            if row < 7:
                neighbors[neighbor_count] = idx + 8
                neighbor_count += 1
            if col > 0:
                neighbors[neighbor_count] = idx - 1
                neighbor_count += 1
            if col < 7:
                neighbors[neighbor_count] = idx + 1
                neighbor_count += 1

            for n in range(neighbor_count):
                next_idx = neighbors[n]
                next_bit = np.uint64(1) << np.uint64(next_idx)
                if (u_board & next_bit) == np.uint64(0) and (visited & next_bit) == np.uint64(0):
                    visited |= next_bit
                    stack[stack_size] = next_idx
                    stack_size += 1

        if region_size > largest_region:
            largest_region = region_size
        if region_size <= 3:
            small_regions += 1

    return largest_region, small_regions, region_count
```

- [ ] **Step 2: Add trap penalty scoring**

Below the region function, add:

```python
@njit(cache=True)
def calculate_trap_penalty_jit(board_mask: int) -> int:
    u_board = np.uint64(board_mask)
    traps = 0

    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            bit = np.uint64(1) << np.uint64(idx)
            if (u_board & bit) != np.uint64(0):
                continue

            blocked_neighbors = 0
            open_neighbors = 0

            if r == 0 or (u_board & (np.uint64(1) << np.uint64(idx - 8))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1
            if r == 7 or (u_board & (np.uint64(1) << np.uint64(idx + 8))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1
            if c == 0 or (u_board & (np.uint64(1) << np.uint64(idx - 1))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1
            if c == 7 or (u_board & (np.uint64(1) << np.uint64(idx + 1))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1

            if blocked_neighbors >= 3:
                traps += 2
            elif open_neighbors == 1:
                traps += 1

    return traps
```

- [ ] **Step 3: Compile-check solver**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/modules/solver.py
```

Expected: command exits with code 0.

---

### Task 5: Add Line Readiness Scoring

**Files:**
- Modify: `block_blast_solver/modules/solver.py`

- [ ] **Step 1: Add line readiness function**

Below `calculate_trap_penalty_jit`, add:

```python
@njit(cache=True)
def calculate_line_readiness_survival_jit(board_mask: int) -> int:
    u_board = np.uint64(board_mask)
    readiness = 0

    for r in range(8):
        row_mask = np.uint64(0xFF) << np.uint64(r * 8)
        filled = popcount(int(u_board & row_mask))
        missing = 8 - filled
        if missing == 1:
            readiness += 6
        elif missing == 2:
            readiness += 3
        elif missing == 3:
            readiness += 1

    for c in range(8):
        col_mask = np.uint64(0x0101010101010101) << np.uint64(c)
        filled = popcount(int(u_board & col_mask))
        missing = 8 - filled
        if missing == 1:
            readiness += 6
        elif missing == 2:
            readiness += 3
        elif missing == 3:
            readiness += 1

    return readiness
```

- [ ] **Step 2: Compile-check solver**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/modules/solver.py
```

Expected: command exits with code 0.

---

### Task 6: Replace Final Evaluation With Survival Evaluator

**Files:**
- Modify: `block_blast_solver/modules/solver.py`

- [ ] **Step 1: Add survival evaluator function**

Below `check_readiness_jit`, add:

```python
@njit(cache=True)
def evaluate_survival_score_jit(board_mask: int,
                                accumulated_score: float,
                                w_clear: float,
                                w_empty: float,
                                w_holes: float,
                                w_bumpiness: float,
                                w_readiness: float,
                                w_future_fits: float,
                                w_largest_region: float,
                                w_small_region_penalty: float,
                                w_line_readiness_survival: float,
                                w_trap_penalty: float) -> float:
    empty_cells = 64 - popcount(board_mask)
    holes = calculate_holes_jit(board_mask)
    bumpiness = calculate_bumpiness_jit(board_mask)
    readiness = check_readiness_jit(board_mask, empty_cells)
    future_fits = count_future_piece_fits_jit(board_mask)
    largest_region, small_regions, region_count = calculate_empty_regions_jit(board_mask)
    line_readiness = calculate_line_readiness_survival_jit(board_mask)
    traps = calculate_trap_penalty_jit(board_mask)

    fragmentation_penalty = max(0, region_count - 1) * 4

    return (
        accumulated_score
        + (w_clear * 0.35)
        + (w_empty * empty_cells)
        + (w_holes * holes)
        + (w_bumpiness * bumpiness)
        + (w_readiness * readiness)
        + (w_future_fits * future_fits)
        + (w_largest_region * largest_region)
        + (w_small_region_penalty * small_regions)
        + (w_line_readiness_survival * line_readiness)
        + (w_trap_penalty * traps)
        - (fragmentation_penalty * 20.0)
    )
```

- [ ] **Step 2: Update `solve_recursive()` signature**

Add these parameters after `w_readiness: float`:

```python
                    w_future_fits: float,
                    w_largest_region: float,
                    w_small_region_penalty: float,
                    w_line_readiness_survival: float,
                    w_trap_penalty: float) -> None:
```

- [ ] **Step 3: Replace terminal score block**

In `solve_recursive()`, replace the body of `if depth == max_depth:` with:

```python
    if depth == max_depth:
        score = evaluate_survival_score_jit(
            board_mask,
            accumulated_score,
            w_clear,
            w_empty,
            w_holes,
            w_bumpiness,
            w_readiness,
            w_future_fits,
            w_largest_region,
            w_small_region_penalty,
            w_line_readiness_survival,
            w_trap_penalty,
        )

        if score > best_score_holder[0]:
            best_score_holder[0] = score
            for d in range(max_depth):
                best_moves_global[d, 0] = current_moves[d, 0]
                best_moves_global[d, 1] = current_moves[d, 1]
                best_moves_global[d, 2] = current_moves[d, 2]
        return
```

- [ ] **Step 4: Update recursive call parameters**

In the recursive `solve_recursive(...)` call, append:

```python
                            w_future_fits,
                            w_largest_region,
                            w_small_region_penalty,
                            w_line_readiness_survival,
                            w_trap_penalty
```

- [ ] **Step 5: Update top-level `solve_recursive(...)` call in `solve()`**

Append these config values after `config.W_READINESS`:

```python
        config.W_FUTURE_FITS,
        config.W_LARGEST_REGION,
        config.W_SMALL_REGION_PENALTY,
        config.W_LINE_READINESS_SURVIVAL,
        config.W_TRAP_PENALTY
```

- [ ] **Step 6: Compile-check solver**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/modules/solver.py
```

Expected: command exits with code 0.

---

### Task 7: Run Tests and Tune Weights

**Files:**
- Modify: `block_blast_solver/config.py` if test results require weight adjustment
- Modify: `block_blast_solver/modules/solver.py` only if a scoring feature has a logic error

- [ ] **Step 1: Run deterministic tests**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_survival -v
```

Expected: all tests pass.

- [ ] **Step 2: If `test_preserves_open_three_by_three_space` fails**

Increase future-fit and largest-region weights in `config.py`:

```python
W_FUTURE_FITS = 55.0
W_LARGEST_REGION = 24.0
```

Then rerun:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_survival -v
```

Expected: all tests pass.

- [ ] **Step 3: If `test_prefers_clear_when_board_is_constrained` fails**

Increase line readiness weight in `config.py`:

```python
W_LINE_READINESS_SURVIVAL = 40.0
```

Then rerun:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m unittest tests.test_solver_survival -v
```

Expected: all tests pass.

---

### Task 8: Runtime and Live Smoke Test

**Files:**
- No planned file changes unless verification exposes a bug

- [ ] **Step 1: Run a direct solver smoke test**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -c "import sys, time; sys.path.insert(0, 'block_blast_solver'); import numpy as np; from modules import solver; board=np.zeros((8,8), dtype=np.uint8); board[0,0:4]=1; board[3:7,3]=1; pieces=[np.array([[1,1,1]], dtype=np.uint8), np.array([[1],[1],[1]], dtype=np.uint8), np.array([[1,1],[1,0]], dtype=np.uint8)]; start=time.time(); moves, score=solver.solve(board, pieces); print(moves, score, round((time.time()-start)*1000, 1))"
```

Expected: prints a non-empty move list, a finite score greater than `-1e8`, and runtime. First run may include Numba compile time.

- [ ] **Step 2: Run the command again for warm runtime**

Run the same command again.

Expected: warm runtime is below 50 ms on a normal board.

- [ ] **Step 3: Run live solver**

Run:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe c:/Users/Valo/Desktop/Desktop/coding/blockblast/block_blast_solver/main.py
```

Expected: HUD opens, suggestions appear after stable detection, and recommended placements avoid obviously occupied cells.

- [ ] **Step 4: Final diagnostics**

Run VS Code diagnostics or compile checks:

```powershell
& C:/Users/Valo/AppData/Local/Programs/Python/Python313/python.exe -m py_compile block_blast_solver/config.py block_blast_solver/main.py block_blast_solver/modules/solver.py
```

Expected: command exits with code 0.

---

## Self-Review

Spec coverage:

- Future piece fit count: Task 3 and Task 6.
- Largest empty region: Task 4 and Task 6.
- Line readiness: Task 5 and Task 6.
- Trap penalties: Task 4 and Task 6.
- Immediate clears remain but less dominant: Task 6.
- Configurable weights: Task 2 and Task 7.
- Deterministic tests: Task 1 and Task 7.
- Runtime verification: Task 8.

Placeholder scan: no TBD/TODO/implement-later placeholders remain.

Type consistency: all new functions use `int`, `float`, `np.ndarray`, and Numba-compatible scalar/array types consistent with existing `solver.py` patterns.
