# Learned Block Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace classical brightness/LAB block heuristics with an ONNX board-cell classifier and inventory-slot masker so themed / mixed-color Block Blast skins still yield correct binary 8×8 boards and piece matrices.

**Architecture:** Keep `BOARD_ROI` / `PIECES_ROI` geometry. Add `ModelRegistry` that lazy-loads two ONNX models (32×32 cell classifier, 128×128 slot U-Net-lite masker). `get_board_state` / `get_pieces` use models when available and fall back to existing heuristics otherwise. Screenshot fixtures under `tests/fixtures/vision_skins/` drive training and regression.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, `onnxruntime==1.27.1` (runtime), PyTorch (train extra only), unittest + coverage, ruff.

## Global Constraints

- Board is always **8×8**; outputs stay **binary only** (no color API).
- Public signatures stay: `get_board_state(frame) -> Tuple[np.ndarray, bool]` and `get_pieces(frame, cell_w, cell_h) -> List[Optional[np.ndarray]]`.
- Runtime must not require PyTorch; Windows-only packages (`dxcam`, `PyGetWindow`) stay lazy / not installed on Linux.
- Invoke tools via `.venv/bin/...` after `pip install -e ".[dev]"` (and `.[train]` only when training).
- Lint: `.venv/bin/ruff check .` — Test: `.venv/bin/coverage run -m unittest discover -s tests -v` — Coverage gate: `.venv/bin/coverage report --omit=block_blast_solver/modules/solver.py --fail-under=70`.
- Do not retrain inside CI; commit small ONNX weights + fixtures.
- Spec: `docs/superpowers/specs/2026-07-16-learned-block-detection-design.md`.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `block_blast_solver/config.py` | Add model paths, thresholds, `VISION_FORCE_HEURISTIC` |
| `block_blast_solver/modules/vision_models.py` | `ModelRegistry`, ONNX cell classifier + slot masker wrappers |
| `block_blast_solver/modules/vision.py` | Wire learned paths; keep heuristic functions as fallback |
| `block_blast_solver/models/board_cell_classifier.onnx` | Committed board weights |
| `block_blast_solver/models/inventory_slot_masker.onnx` | Committed inventory weights |
| `block_blast_solver/models/__init__.py` | Empty package marker so setuptools includes models |
| `tests/fixtures/vision_skins/*.png` + `*.json` | Screenshot + label pairs |
| `tests/test_vision_fixture_schema.py` | JSON schema / loader validation |
| `tests/test_vision_models.py` | Registry load/fallback/occlusion helpers |
| `tests/test_vision_board.py` | Keep; add force-heuristic coverage as needed |
| `tests/test_vision_pieces.py` | Keep heuristic synthetic coverage |
| `tests/test_vision_skins.py` | End-to-end learned path on real fixtures |
| `scripts/export_vision_training_data.py` | Crops/masks from fixtures → training dataset |
| `scripts/train_vision_models.py` | Train + export ONNX; gate on fixture accuracy |
| `pyproject.toml` / `requirements*.txt` | `onnxruntime` runtime; `torch` in `train` extra |
| `README.md` / `AGENTS.md` | Note learned vision + train extra |

---

### Task 1: Skin fixture schema, loader, and labeled screenshots

**Files:**
- Create: `tests/fixtures/vision_skins/README.md`
- Create: `block_blast_solver/modules/vision_fixtures.py` (test-shared loader; importable from tests)
- Create: `tests/test_vision_fixture_schema.py`
- Create: `tests/fixtures/vision_skins/<skin>.png` + `<skin>.json` for each brainstorm screenshot
- Modify: `docs/superpowers/specs/2026-07-16-learned-block-detection-design.md` (status already Approved)

**Interfaces:**
- Consumes: brainstorm screenshots attached in the design conversation (save into the fixture dir)
- Produces:
  - `load_skin_fixture(path: Path) -> SkinFixture` dataclass with fields `image: np.ndarray` (BGR), `board: np.ndarray` shape `(8,8)` uint8, `pieces: list[Optional[np.ndarray]]` length 3, `board_roi: list[float]`, `pieces_roi: list[float]`
  - `iter_skin_fixtures(root: Path | None = None) -> Iterator[SkinFixture]`
  - `validate_skin_label(data: dict) -> None` raises `ValueError` on bad schema

- [ ] **Step 1: Write the failing schema test**

Create `tests/test_vision_fixture_schema.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from block_blast_solver.modules.vision_fixtures import load_skin_fixture, validate_skin_label


class VisionFixtureSchemaTests(unittest.TestCase):
    def test_validate_accepts_minimal_label(self):
        label = {
            "board_roi": [0.1, 0.2, 0.8, 0.5],
            "pieces_roi": [0.1, 0.75, 0.8, 0.2],
            "board": [[0] * 8 for _ in range(8)],
            "pieces": [None, [[1, 1], [1, 1]], None],
        }
        validate_skin_label(label)

    def test_validate_rejects_wrong_board_shape(self):
        label = {
            "board_roi": [0.1, 0.2, 0.8, 0.5],
            "pieces_roi": [0.1, 0.75, 0.8, 0.2],
            "board": [[0] * 7 for _ in range(8)],
            "pieces": [None, None, None],
        }
        with self.assertRaises(ValueError):
            validate_skin_label(label)

    def test_load_skin_fixture_round_trip(self):
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "demo.png"
            label_path = root / "demo.json"
            frame = np.zeros((200, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), frame)
            label = {
                "board_roi": [0.1, 0.1, 0.8, 0.6],
                "pieces_roi": [0.1, 0.75, 0.8, 0.2],
                "board": [[0] * 8 for _ in range(8)],
                "pieces": [[[1]], None, None],
            }
            label_path.write_text(json.dumps(label), encoding="utf-8")
            fixture = load_skin_fixture(label_path)
            self.assertEqual(fixture.image.shape, (200, 100, 3))
            self.assertEqual(fixture.board.shape, (8, 8))
            self.assertEqual(fixture.pieces[0].tolist(), [[1]])
            self.assertIsNone(fixture.pieces[1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_vision_fixture_schema -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'block_blast_solver.modules.vision_fixtures'`

- [ ] **Step 3: Implement loader + schema validation**

Create `block_blast_solver/modules/vision_fixtures.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

DEFAULT_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "vision_skins"


@dataclass(frozen=True)
class SkinFixture:
    name: str
    image: np.ndarray
    board: np.ndarray
    pieces: List[Optional[np.ndarray]]
    board_roi: list[float]
    pieces_roi: list[float]
    label_path: Path
    image_path: Path


def validate_skin_label(data: object) -> None:
    if not isinstance(data, dict):
        raise ValueError("label root must be an object")
    for key in ("board_roi", "pieces_roi", "board", "pieces"):
        if key not in data:
            raise ValueError(f"missing key: {key}")
    for roi_key in ("board_roi", "pieces_roi"):
        roi = data[roi_key]
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError(f"{roi_key} must be 4 floats")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in roi):
            raise ValueError(f"{roi_key} values must be numbers")
    board = data["board"]
    if not isinstance(board, list) or len(board) != 8:
        raise ValueError("board must be 8 rows")
    for row in board:
        if not isinstance(row, list) or len(row) != 8:
            raise ValueError("board rows must have 8 cells")
        if any(cell not in (0, 1) for cell in row):
            raise ValueError("board cells must be 0 or 1")
    pieces = data["pieces"]
    if not isinstance(pieces, list) or len(pieces) != 3:
        raise ValueError("pieces must be a list of length 3")
    for piece in pieces:
        if piece is None:
            continue
        if not isinstance(piece, list) or not piece:
            raise ValueError("piece matrix must be a non-empty 2D list")
        width = len(piece[0])
        if width < 1:
            raise ValueError("piece matrix has empty row")
        for row in piece:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError("piece rows must share width")
            if any(cell not in (0, 1) for cell in row):
                raise ValueError("piece cells must be 0 or 1")


def load_skin_fixture(label_path: Path) -> SkinFixture:
    label_path = Path(label_path)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    validate_skin_label(data)
    image_path = label_path.with_suffix(".png")
    if not image_path.exists():
        image_path = label_path.with_suffix(".jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"no image beside {label_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    pieces: List[Optional[np.ndarray]] = []
    for piece in data["pieces"]:
        if piece is None:
            pieces.append(None)
        else:
            pieces.append(np.asarray(piece, dtype=np.uint8))
    return SkinFixture(
        name=label_path.stem,
        image=image,
        board=np.asarray(data["board"], dtype=np.uint8),
        pieces=pieces,
        board_roi=[float(v) for v in data["board_roi"]],
        pieces_roi=[float(v) for v in data["pieces_roi"]],
        label_path=label_path,
        image_path=image_path,
    )


def iter_skin_fixtures(root: Path | None = None) -> Iterator[SkinFixture]:
    fixture_root = Path(root) if root is not None else DEFAULT_FIXTURE_ROOT
    for label_path in sorted(fixture_root.glob("*.json")):
        yield load_skin_fixture(label_path)
```

- [ ] **Step 4: Run schema tests to verify they pass**

Run: `.venv/bin/python -m unittest tests.test_vision_fixture_schema -v`

Expected: PASS

- [ ] **Step 5: Check in screenshots + hand-authored labels**

1. Save each brainstorm screenshot into `tests/fixtures/vision_skins/` with stable names, e.g.:
   - `pearl_mixed.png`
   - `gem_busy.png`
   - `gem_five_line.png`
   - `cake_wafer.png`
   - `gem_two_rows.png`
   - `watermelon_purple.png`
   - `heart_pastel.png`
   - `watermelon_green.png`
   - `wood_classic.png`
   - `stone_corner.png`
2. For each image, measure pixel ROIs for the 8×8 board and the three-piece inventory strip; convert to normalized `[x, y, w, h]`.
3. Hand-author matching `<name>.json` with exact `board` and `pieces` matrices from visual inspection (empty slots = `null`).
4. Write `tests/fixtures/vision_skins/README.md` describing the schema and naming.
5. Smoke-load all fixtures:

```bash
.venv/bin/python -c "from block_blast_solver.modules.vision_fixtures import iter_skin_fixtures; print([f.name for f in iter_skin_fixtures()])"
```

Expected: prints every stem; no exceptions.

- [ ] **Step 6: Commit**

```bash
git add block_blast_solver/modules/vision_fixtures.py tests/test_vision_fixture_schema.py tests/fixtures/vision_skins docs/superpowers/specs/2026-07-16-learned-block-detection-design.md
git add -f docs/superpowers/specs/2026-07-16-learned-block-detection-design.md
git commit -m "Add vision skin fixture schema and labeled screenshots"
```

---

### Task 2: Config knobs + ModelRegistry with heuristic fallback

**Files:**
- Modify: `block_blast_solver/config.py`
- Create: `block_blast_solver/modules/vision_models.py`
- Create: `block_blast_solver/models/__init__.py`
- Create: `tests/test_vision_models.py`
- Modify: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`

**Interfaces:**
- Consumes: none from Task 1 at runtime
- Produces:
  - Config: `BOARD_CELL_MODEL_PATH`, `INVENTORY_MASK_MODEL_PATH`, `T_BOARD=0.5`, `T_MASK=0.5`, `T_CELL=0.30`, `VISION_FORCE_HEURISTIC` (bool from env `VISION_FORCE_HEURISTIC`)
  - `class BoardCellClassifier`: `predict_proba(crop_bgr: np.ndarray) -> float`
  - `class InventorySlotMasker`: `predict_mask(slot_bgr: np.ndarray) -> np.ndarray`  # float32 HxW in [0,1] at slot resolution
  - `class ModelRegistry`: `get() -> ModelRegistry`, properties `board_classifier: BoardCellClassifier | None`, `inventory_masker: InventorySlotMasker | None`, `using_learned: bool`, `reset_for_tests() -> None`
  - Package dep: `onnxruntime==1.27.1`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_vision_models.py`:

```python
import os
import unittest
from unittest import mock

import numpy as np

from block_blast_solver.modules import vision_models


class VisionModelsTests(unittest.TestCase):
    def setUp(self):
        vision_models.ModelRegistry.reset_for_tests()

    def tearDown(self):
        vision_models.ModelRegistry.reset_for_tests()
        os.environ.pop("VISION_FORCE_HEURISTIC", None)

    def test_force_heuristic_disables_models(self):
        os.environ["VISION_FORCE_HEURISTIC"] = "1"
        vision_models.ModelRegistry.reset_for_tests()
        registry = vision_models.ModelRegistry.get()
        self.assertFalse(registry.using_learned)
        self.assertIsNone(registry.board_classifier)
        self.assertIsNone(registry.inventory_masker)

    def test_missing_weights_fall_back(self):
        with mock.patch.object(vision_models, "DEFAULT_BOARD_MODEL", "/no/such/board.onnx"), mock.patch.object(
            vision_models, "DEFAULT_MASK_MODEL", "/no/such/mask.onnx"
        ):
            vision_models.ModelRegistry.reset_for_tests()
            registry = vision_models.ModelRegistry.get()
        self.assertFalse(registry.using_learned)

    def test_occlusion_helper_flags_uncertain_board(self):
        probs = np.full((8, 8), 0.5, dtype=np.float32)
        self.assertTrue(vision_models.board_probs_are_occluded(probs))
        empty = np.full((8, 8), 0.05, dtype=np.float32)
        self.assertFalse(vision_models.board_probs_are_occluded(empty))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_vision_models -v`

Expected: FAIL (`No module named ...vision_models` or missing attributes)

- [ ] **Step 3: Add config + install onnxruntime + implement registry**

In `block_blast_solver/config.py` add:

```python
import os
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent / "models"
BOARD_CELL_MODEL_PATH = os.environ.get(
    "BLOCK_BLAST_BOARD_CELL_MODEL",
    str(_MODELS_DIR / "board_cell_classifier.onnx"),
)
INVENTORY_MASK_MODEL_PATH = os.environ.get(
    "BLOCK_BLAST_INVENTORY_MASK_MODEL",
    str(_MODELS_DIR / "inventory_slot_masker.onnx"),
)
T_BOARD = 0.5
T_MASK = 0.5
T_CELL = 0.30

def vision_force_heuristic() -> bool:
    return os.environ.get("VISION_FORCE_HEURISTIC", "").strip().lower() in {"1", "true", "yes", "on"}
```

Create empty `block_blast_solver/models/__init__.py`.

Update `pyproject.toml` dependencies to include `"onnxruntime==1.27.1"`, add:

```toml
train = [
    "torch==2.7.1",
]
```

and package data:

```toml
[tool.setuptools.package-data]
block_blast_solver = ["models/*.onnx"]
```

Mirror `onnxruntime==1.27.1` into `requirements.txt`. Add `torch==2.7.1` note under train in README later.

Install: `.venv/bin/python -m pip install -e ".[dev]"`

Create `block_blast_solver/modules/vision_models.py` implementing:

- `DEFAULT_BOARD_MODEL` / `DEFAULT_MASK_MODEL` from config paths
- `board_probs_are_occluded(probs: np.ndarray) -> bool` → True if count of cells with `abs(p-0.5) < 0.15` is `> 16`
- Lazy ONNX session load; on any failure set classifiers to `None` and log once
- `BoardCellClassifier.predict_proba`: resize crop to 32×32 RGB float32 NCHW `/255`, run session, return scalar sigmoid/prob
- `InventorySlotMasker.predict_mask`: resize to 128×128, run session, upsample mask to original HxW with `cv2.resize(..., INTER_LINEAR)`
- Until real ONNX exists, missing files → `using_learned is False` (tests above must pass)

- [ ] **Step 4: Run registry tests**

Run: `.venv/bin/python -m unittest tests.test_vision_models -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add block_blast_solver/config.py block_blast_solver/modules/vision_models.py block_blast_solver/models/__init__.py tests/test_vision_models.py pyproject.toml requirements.txt
git commit -m "Add ONNX vision ModelRegistry and config thresholds"
```

---

### Task 3: Wire learned board detection into `get_board_state`

**Files:**
- Modify: `block_blast_solver/modules/vision.py`
- Modify: `tests/test_vision_board.py`
- Modify: `tests/test_vision_models.py` (optional inject test)

**Interfaces:**
- Consumes: `ModelRegistry.get().board_classifier`, `config.T_BOARD`, `board_probs_are_occluded`
- Produces: `get_board_state` uses learned path when `registry.board_classifier is not None`; else existing heuristic. Keep `_get_board_state_heuristic(frame)` as the renamed current body.

- [ ] **Step 1: Write failing test for learned board path with a fake classifier**

Add to `tests/test_vision_board.py`:

```python
    def test_learned_path_uses_classifier_probabilities(self):
        from block_blast_solver.modules import vision_models

        class FakeClassifier:
            def predict_proba(self, crop_bgr):
                # Bright crops → occupied in the synthetic draw_board helper
                return float(np.mean(crop_bgr) / 255.0)

        vision_models.ModelRegistry.reset_for_tests()
        registry = vision_models.ModelRegistry.get()
        registry.board_classifier = FakeClassifier()
        registry.inventory_masker = None
        registry.using_learned = True

        expected_cells = {(0, 0), (3, 4), (7, 7)}
        board, occluded = vision.get_board_state(draw_board(expected_cells))
        self.assertFalse(occluded)
        self.assertEqual({tuple(cell) for cell in np.argwhere(board == 1)}, expected_cells)
        vision_models.ModelRegistry.reset_for_tests()
```

Also assert existing heuristic tests still pass with `VISION_FORCE_HEURISTIC=1`.

- [ ] **Step 2: Run the new test — expect FAIL** until wiring exists

Run: `.venv/bin/python -m unittest tests.TestVisionBoardTests.test_learned_path_uses_classifier_probabilities -v`  
(Adjust class path to `tests.test_vision_board.VisionBoardTests.test_learned_path_uses_classifier_probabilities`)

Expected: FAIL (classifier ignored / wrong board) or error if attribute assignment not supported — implement registry fields as settable instance attrs.

- [ ] **Step 3: Implement learned board path**

In `vision.py`:

1. Rename current `get_board_state` body to `_get_board_state_heuristic(frame)`.
2. New `get_board_state`:
   - If `config.vision_force_heuristic()` or no classifier → heuristic
   - Else crop board ROI, loop 8×8 inner 60% crops, `predict_proba` each, build `probs` array
   - On any exception → return `zeros, True`
   - If `board_probs_are_occluded(probs)` → `zeros, True`
   - Else `board = (probs >= config.T_BOARD).astype(uint8)`, `occluded=False`

Keep cell geometry identical to today’s loop (offsets 0.20 / 0.60).

- [ ] **Step 4: Run board tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_vision_board -v
VISION_FORCE_HEURISTIC=1 .venv/bin/python -m unittest tests.test_vision_board -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add block_blast_solver/modules/vision.py tests/test_vision_board.py
git commit -m "Wire ONNX board cell classifier into get_board_state"
```

---

### Task 4: Wire inventory masker + mask-based piece decode

**Files:**
- Modify: `block_blast_solver/modules/vision.py`
- Modify: `tests/test_vision_pieces.py`

**Interfaces:**
- Consumes: `InventorySlotMasker.predict_mask`, `config.T_MASK`, `config.T_CELL`, existing `_estimate_piece_grid_dims`, `_trim_empty_edges`
- Produces: `get_pieces` learned path; `_get_pieces_heuristic(...)` = current body; `_piece_from_mask(mask, inv_cell_w, inv_cell_h) -> Optional[np.ndarray]`

- [ ] **Step 1: Write failing test with fake masker**

Add a test that builds a slot-sized mask for an L piece and injects:

```python
class FakeMasker:
    def __init__(self, masks_by_slot):
        self.masks_by_slot = masks_by_slot
        self.calls = 0

    def predict_mask(self, slot_bgr):
        mask = self.masks_by_slot[self.calls]
        self.calls += 1
        return mask
```

Construct three float masks matching known shapes (paint 1.0 on block cells, 0.0 elsewhere at slot resolution), set them on the registry, call `get_pieces`, assert matrices match. Exact mask painting can follow `draw_inventory` geometry from `tests/test_vision_pieces.py`.

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `_piece_from_mask` + learned `get_pieces`**

Logic:

1. `mask_bin = (mask >= T_MASK).astype(uint8) * 255`
2. If `cv2.countNonZero(mask_bin) < min_block_area` → `None`
3. `bbox = cv2.boundingRect(cv2.findNonZero(mask_bin))`
4. `num_rows, num_cols = _estimate_piece_grid_dims(...)`
5. For each subcell, if mean(mask_bin inner)/255 > `T_CELL` → 1
6. `_trim_empty_edges`; return matrix or `None` if empty

`get_pieces`: if masker present, per-slot `predict_mask` → `_piece_from_mask`; on exception → `[None,None,None]`; else heuristic.

- [ ] **Step 4: Run piece + board tests**

```bash
.venv/bin/python -m unittest tests.test_vision_pieces tests.test_vision_board -v
VISION_FORCE_HEURISTIC=1 .venv/bin/python -m unittest tests.test_vision_pieces -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add block_blast_solver/modules/vision.py tests/test_vision_pieces.py
git commit -m "Wire inventory slot masker into get_pieces"
```

---

### Task 5: Training data export script

**Files:**
- Create: `scripts/export_vision_training_data.py`
- Create: `tests/test_export_vision_training_data.py` (optional light test on temp fixture)

**Interfaces:**
- Consumes: `iter_skin_fixtures`, OpenCV
- Produces: directory layout:

```text
datasets/vision/
  board_cells/{occupied,empty}/*.png
  inventory/{images,masks}/*.png
  manifest.json
```

Board cells: for each fixture cell, save inner 60% crop labeled by `board[r,c]`.  
Inventory: for each non-null piece, rasterize the binary matrix into a slot-sized mask aligned to centered piece cells (use same block scale heuristic as tests: scale piece to ~47% of board cell using fixture image board width/8), save slot RGB crop + mask PNG (0/255).

- [ ] **Step 1: Implement export script** with CLI:

```bash
.venv/bin/python scripts/export_vision_training_data.py --out datasets/vision
```

Refuse to run if `iter_skin_fixtures()` is empty.

- [ ] **Step 2: Run export and spot-check counts**

```bash
.venv/bin/python scripts/export_vision_training_data.py --out datasets/vision
.venv/bin/python -c "from pathlib import Path; print(sum(1 for _ in Path('datasets/vision/board_cells/occupied').glob('*.png')), sum(1 for _ in Path('datasets/vision/board_cells/empty').glob('*.png')))"
```

Expected: occupied + empty == `8*8*num_fixtures`; inventory image/mask counts equal.

- [ ] **Step 3: Add `datasets/` to `.gitignore`**

- [ ] **Step 4: Commit** (script + gitignore only; not dataset blobs)

```bash
git add scripts/export_vision_training_data.py .gitignore
git commit -m "Add vision training data export from skin fixtures"
```

---

### Task 6: Train board classifier + inventory masker and commit ONNX

**Files:**
- Create: `scripts/train_vision_models.py`
- Create: `block_blast_solver/models/board_cell_classifier.onnx`
- Create: `block_blast_solver/models/inventory_slot_masker.onnx`
- Modify: `pyproject.toml` (`train` extra already added)

**Interfaces:**
- Consumes: `datasets/vision` from Task 5; `torch` from `.[train]`
- Produces: ONNX files with:
  - Board: input `input` float32 `[N,3,32,32]`, output `prob` float32 `[N,1]`
  - Mask: input `input` float32 `[N,3,128,128]`, output `prob` float32 `[N,1,128,128]`
- Acceptance gate before writing ONNX: with exported weights loaded through `ModelRegistry`, every skin fixture must match labeled `board` and `pieces` exactly via `get_board_state` / `get_pieces`. If any fail, exit non-zero and do not overwrite committed ONNX.

- [ ] **Step 1: Install train extra**

```bash
.venv/bin/python -m pip install -e ".[dev,train]"
```

- [ ] **Step 2: Implement tiny models + training loop in `scripts/train_vision_models.py`**

Board CNN (example):

```python
class BoardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)  # logits
```

Inventory U-Net-lite: 3→16→32 encoder, bottleneck, decoder with skips, final `Conv2d → 1` logits.

Train with BCEWithLogits, Adam, CPU OK, few epochs; augment lightly (brightness/flip for board cells only if labels remain valid — prefer no geometric aug on inventory masks beyond small noise).

Export ONNX with `torch.onnx.export(..., opset_version=17, input_names=["input"], output_names=["prob"])`. Classifier wrapper must apply sigmoid to logits if ONNX exports logits — pick one convention and match `BoardCellClassifier` / `InventorySlotMasker` preprocessing/postprocessing exactly (document in script docstring: **ONNX outputs probabilities in [0,1]** via exported `sigmoid`).

- [ ] **Step 3: Train, gate, write weights**

```bash
.venv/bin/python scripts/export_vision_training_data.py --out datasets/vision
.venv/bin/python scripts/train_vision_models.py --data datasets/vision --out-dir block_blast_solver/models
```

Expected: script prints fixture-by-fixture PASS and writes both `.onnx` files.

If gate fails: improve labels/export alignment or train longer; do not commit failing weights.

- [ ] **Step 4: Smoke-load via registry**

```bash
.venv/bin/python -c "from block_blast_solver.modules.vision_models import ModelRegistry; ModelRegistry.reset_for_tests(); r=ModelRegistry.get(); print(r.using_learned, r.board_classifier is not None, r.inventory_masker is not None)"
```

Expected: `True True True`

- [ ] **Step 5: Commit weights + train script**

```bash
git add scripts/train_vision_models.py block_blast_solver/models/*.onnx pyproject.toml
git commit -m "Train and commit ONNX board classifier and inventory masker"
```

---

### Task 7: End-to-end skin regression tests + docs

**Files:**
- Create: `tests/test_vision_skins.py`
- Modify: `README.md`, `AGENTS.md`, `CONTRIBUTING.md` (brief notes)
- Modify: `requirements-dev.txt` if needed (still `-r requirements.txt` which includes onnxruntime)

**Interfaces:**
- Consumes: committed ONNX + fixtures
- Produces: CI-covered regression that fails if skins regress

- [ ] **Step 1: Write skin regression tests**

```python
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
```

- [ ] **Step 2: Run full suite + lint + coverage gate**

```bash
.venv/bin/ruff check .
.venv/bin/coverage run -m unittest discover -s tests -v
.venv/bin/coverage report
.venv/bin/coverage report --omit=block_blast_solver/modules/solver.py --fail-under=70
```

Expected: all PASS; coverage gate PASS.

- [ ] **Step 3: Document**

In `README.md` Architecture / Features: mention ONNX skin-robust vision, `VISION_FORCE_HEURISTIC`, and `pip install -e ".[train]"` for retraining.  
In `AGENTS.md`: note onnxruntime is required; torch only for train; fixtures path.

- [ ] **Step 4: Commit**

```bash
git add tests/test_vision_skins.py README.md AGENTS.md CONTRIBUTING.md
git commit -m "Add skin regression tests and document learned vision"
```

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| Board cell ONNX classifier 32×32 | 2, 3, 6 |
| Inventory U-Net-lite masker 128×128 + upsample | 2, 4, 6 |
| Classical piece decode from mask | 4 |
| ModelRegistry + heuristic fallback + `VISION_FORCE_HEURISTIC` | 2, 3, 4 |
| Screenshot fixtures + JSON labels | 1 |
| Training export + train + accuracy gate | 5, 6 |
| Runtime onnxruntime / train-only torch | 2, 6 |
| E2E skin tests + CI weights | 6, 7 |
| Binary-only 8×8 API unchanged | 3, 4, 7 |
| Occlusion rules (>16 uncertain cells) | 2, 3 |
| Fail-closed on inference errors | 3, 4 |

## Placeholder / consistency check

- Threshold names: `T_BOARD`, `T_MASK`, `T_CELL` used consistently.
- ONNX I/O names: `input` / `prob`; probabilities in `[0,1]` after sigmoid in export.
- Registry API: `board_classifier`, `inventory_masker`, `using_learned`, `reset_for_tests()` used across tasks.
- No TBD/TODO left in steps.
