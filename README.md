# Block Blast Solver

A Windows desktop assistant that reads a mirrored Block Blast game, detects the board and inventory, evaluates placement orders, and draws suggested moves in a live HUD.

## Features

- Low-latency, multi-monitor window capture through `dxcam`
- OpenCV board and inventory detection with stable-frame filtering
- Numba-accelerated bitboard search
- Combo, board-survival, and deterministic future-set evaluation
- HUD diagnostics for risk, survival, clear routes, and move order
- Headless capture backend and synthetic vision tests for development

## Requirements

- Python 3.11 or newer (Python 3.13 recommended)
- Windows 10/11 for live capture and calibration
- A mirrored game window matching `WINDOW_TITLE` in [`config.py`](block_blast_solver/config.py)

## Install

Create a virtual environment and install the Windows runtime:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[windows]"
```

Exact direct dependency versions are also recorded in [`requirements.txt`](requirements.txt).

## Run

```bash
block-blast-solver
```

Alternatively:

```bash
python -m block_blast_solver.main
```

On first launch, select the 8×8 board followed by the complete three-piece inventory area. Calibration is stored under the current user's configuration directory. Set `BLOCK_BLAST_CALIBRATION_FILE` to override its location.

Controls:

- `c`: recalibrate
- `q`: quit

## Development

Linux and macOS can run solver, state, vision, and HUD tests, but live capture remains Windows-only.

```bash
python -m pip install -e ".[dev]"
ruff check .
coverage run -m unittest discover -s tests -v
coverage report
coverage report --omit=block_blast_solver/modules/solver.py --fail-under=70
```

Measure warm solver latency with:

```bash
python scripts/benchmark_solver.py --iterations 10
```

`SEARCH_NODE_BUDGET` in [`config.py`](block_blast_solver/config.py) bounds deterministic search work and returns the best completed result when exhausted. Set it to `0` for exhaustive search.

## Architecture

1. `capture.py` captures the target window and maps calibrated regions.
2. `vision.py` detects the 8×8 board and three inventory pieces.
3. `state.py` waits for consecutive matching detections.
4. `solver.py` searches current moves and evaluates representative future sets.
5. `visualizer.py` renders move overlays and diagnostics.

GitHub Actions runs lint, coverage, and tests on Python 3.11/3.13 Linux plus a Python 3.13 Windows capture-import smoke test.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for platform notes and the pull-request checklist.

This project is released under the [MIT License](LICENSE).
