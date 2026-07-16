# Learned Block Detection Design

**Date:** 2026-07-16  
**Status:** Approved — implementation plan next  
**Scope:** Improve board and inventory block detection across themed skins (gem, cake, watermelon, wood, pearlescent, mixed-color pieces, etc.) using a learned detector, while keeping binary-only outputs and the existing 8×8 solver contract.

## Problem

`vision.py` currently decides occupancy with classical cues:

- **Board:** HSV Value brightness clustering (`get_board_state`)
- **Inventory:** LAB distance from slot background + saturation (`get_pieces`), then bbox → grid → fill

Real Block Blast skins break those assumptions: faceted gems, cake/wafer textures, watermelon seeds/rind, stone/wood grain, pearlescent gradients, drop shadows, and **mixed-color pieces** (multiple hues in one shape). Brightness and single-color distance are not reliable “blockness” signals.

## Goals

- Detect **8×8 board** occupancy and **three inventory piece shapes** reliably across the skins shown in the provided screenshots.
- Output remains **binary only** (occupied vs empty / piece matrices). Colors are an input challenge, not part of the API.
- Keep public contracts: `get_board_state`, `get_pieces`, `state` stability, solver, HUD.
- Validate with **real screenshot regression fixtures** (PNG/JPG + JSON labels).
- Prefer a **learned detector** with ONNX Runtime at inference time.

## Non-goals

- Per-cell or per-piece color output
- Non-8×8 boards
- Retraining inside CI
- Changing solver heuristics or HUD layout
- Requiring PyTorch at app runtime

## Approach (selected)

**Cell classifier + inventory mask model**

Keep calibrated `BOARD_ROI` / `PIECES_ROI`. Replace heuristic “is this a block?” decisions with two small ONNX models; keep classical geometry for grid layout and piece matrix decoding.

## Architecture

```text
frame
  ├─ BOARD_ROI → 8×8 cell crops → ONNX cell classifier → board occupancy
  └─ PIECES_ROI → 3 slot crops → ONNX slot masker → binary mask
                                              → grid estimate + fill → trim → piece matrix
```

- Runtime: **ONNX Runtime (CPU)**
- Training: optional PyTorch tooling under `scripts/` / `tools/` (dev/train extra only)
- Heuristic `vision.py` path remains as **fallback** if models are missing or unloadable

## Components

### 1. `BoardCellClassifier` (ONNX)

- **Input:** 32×32 RGB crop of one board cell, taken from the inner 60% of the cell to avoid grid lines
- **Output:** P(occupied) in `[0, 1]`
- **Decision:** occupied if `p >= T_board` (default `0.5`)
- **Occlusion:** invalid ROI / empty crop → `(zeros, True)`. If more than 16 of 64 cells have `|p - 0.5| < 0.15`, treat as occluded. If every cell has `p < T_board`, return empty board with `occluded=False`.

### 2. `InventorySlotMasker` (ONNX)

- **Input:** one inventory slot crop, resized to 128×128 RGB
- **Output:** single-channel probability map at network resolution, then bilinear-upsampled to the original slot-crop size before thresholding
- **Decision:** binary mask at `T_mask` (default `0.5`)
- Mixed colors and textures are a single foreground class (“block vs background”)
- Architecture: small encoder–decoder (U-Net-lite), one foreground channel

### 3. `PieceShapeDecoder` (classical)

Reuse/adapt existing helpers:

- bbox from mask → `_estimate_piece_grid_dims` (board cell size + `PIECE_SCALE_FACTOR` as prior)
- per-cell fill from mask occupancy ratio (`T_cell`)
- `_trim_empty_edges`
- Keep drop-shadow / phantom-cell gates, now driven by the mask instead of LAB

### 4. `ModelRegistry`

- Lazy-load ONNX sessions once per process
- Default weight paths: `block_blast_solver/models/board_cell_classifier.onnx` and `block_blast_solver/models/inventory_slot_masker.onnx` (overridable via config/env)
- On load failure: log once, fall back to current heuristic path
- `VISION_FORCE_HEURISTIC=1` forces heuristic mode

### 5. Fixture pack

Under `tests/fixtures/vision_skins/`:

- Screenshot images from the provided examples (hand-copied into the repo during implementation)
- Matching hand-authored JSON labels: `board` (8×8 of 0/1), `pieces` (three 0/1 matrices or `null`), normalized `BOARD_ROI` / `PIECES_ROI`
- Label JSON schema is validated by tests before assertions run

### 6. Training scripts (dev-only)

- Export labeled cell crops and slot masks from fixtures
- Train tiny CNN (board) and U-Net-lite (inventory)
- Export ONNX to `block_blast_solver/models/`
- Refuse export if holdout fixture accuracy is below the acceptance bar (exact board + exact piece matrices on the screenshot set)

## Data flow

### Board

1. Validate `BOARD_ROI`; crop board region
2. Split into 8×8 cells; inner crop → resize → classifier → probability
3. `board_state[r,c] = 1` if `p >= T_board`
4. Apply occlusion rules above

### Inventory

1. Validate `PIECES_ROI`; split into three equal slots
2. Slot → masker → probability map → binary mask
3. Mask area below minimum → `None` (empty slot)
4. Else bbox → grid dims → cell fill if mask ratio > `T_cell` → trim → matrix

### Unchanged frame loop

`capture` → vision → `state` stability → solver → HUD

## Error handling

| Case | Behavior |
|------|----------|
| Missing/corrupt ONNX or onnxruntime import failure | Log once; heuristic fallback for the process |
| `VISION_FORCE_HEURISTIC=1` | Skip models |
| Per-frame inference exception | Fail closed for that frame (occluded board / empty pieces); do not crash HUD |
| Empty/noisy inventory mask | `None` for that slot |
| Bad fixture JSON | Fail tests, not runtime |
| Training quality below bar | Do not export/commit ONNX |

## Dependencies

- **Runtime:** add `onnxruntime` (main or appropriate extra)
- **Train/dev only:** `torch` (and related) in a `train` / extended `dev` extra
- Do **not** add Windows-only packages; keep `dxcam` / `PyGetWindow` lazy and out of Linux env

## Testing

### Regression fixtures

Each skin screenshot pairs with JSON labels. Tests assert exact equality of:

- 8×8 `board`
- three `pieces` matrices (or `None`)

### Unit tests

- `test_vision_skins.py` — learned path on all fixtures
- Existing synthetic tests retained for heuristic fallback and ROI fail-closed behavior
- Smoke test: load ONNX once, one forward pass on CPU

### CI

- Commit small ONNX weights + fixtures so Linux CI exercises the learned path without training
- Coverage gate unchanged (`--omit=.../solver.py --fail-under=70`); new wrappers must remain covered

## Configuration (additive)

- Paths to board classifier and inventory masker ONNX files
- Thresholds: `T_board`, `T_mask`, `T_cell`
- `VISION_FORCE_HEURISTIC` env/config flag
- Existing `BOARD_ROI`, `PIECES_ROI`, `PIECE_SCALE_FACTOR`, `DETECTION_STABLE_FRAMES` unchanged in meaning

## Success criteria

- All screenshot fixtures pass exact board + piece matrix assertions with the learned path
- Heuristic fallback still passes existing synthetic vision tests
- App imports and HUD loop remain safe if models are absent (fallback)
- No change to solver input types or 8×8 assumption

## Implementation order (for planning)

1. Fixture schema + check in labeled screenshots
2. ModelRegistry + ONNX wrappers with heuristic fallback
3. Wire board classifier into `get_board_state`
4. Wire inventory masker + PieceShapeDecoder into `get_pieces`
5. Training/export scripts and committed ONNX weights
6. CI + coverage for new modules
