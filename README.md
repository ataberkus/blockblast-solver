# Block Blast Solver

Windows-based Block Blast assistant that captures a mirrored game window, detects the board and inventory pieces, evaluates placements with a combo-aware solver, and renders a live HUD with suggested moves.

## Features

- Window capture via `dxcam`
- Board and inventory piece detection with OpenCV
- Numba-accelerated move search
- Combo/streak-aware scoring to preserve clear chains
- On-screen HUD showing move order and diagnostics

## Requirements

- Windows
- Python 3.13
- A mirrored game window matching `WINDOW_TITLE` in [block_blast_solver/config.py](block_blast_solver/config.py)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python block_blast_solver/main.py
```

On first launch, the app will calibrate the board and pieces regions if `calibration_data.json` is not present.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project Layout

- `block_blast_solver/main.py`: main loop and orchestration
- `block_blast_solver/modules/capture.py`: window capture and calibration
- `block_blast_solver/modules/vision.py`: board and inventory detection
- `block_blast_solver/modules/solver.py`: search and scoring
- `block_blast_solver/modules/visualizer.py`: HUD rendering
- `tests/`: regression tests
