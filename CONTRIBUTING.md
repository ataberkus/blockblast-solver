# Contributing

## Setup

Use Python 3.13 when possible:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

On Windows, install the capture dependencies as well:

```bash
python -m pip install -e ".[dev,windows]"
```

## Quality checks

Run these before opening a pull request:

```bash
ruff check .
coverage run -m unittest discover -s tests -v
coverage report
coverage report --omit=block_blast_solver/modules/solver.py --fail-under=70
```

`solver.py` is included in the full coverage report for visibility, but Numba
`@njit` bodies are mostly untraceable. The 70% gate therefore omits the solver
module and relies on the behavioral solver tests for that path.

Solver changes should include deterministic regression cases. Vision changes should include synthetic images or sanitized fixtures that exercise the affected theme, brightness, scale, or occlusion condition.

## Windows manual checks

The hosted test suite does not launch an interactive DXGI/OpenCV desktop. Changes to live capture or calibration therefore also need a manual check for:

- minimized and restored windows;
- negative monitor coordinates and windows spanning displays;
- 100%, 125%, and 150% display scaling;
- calibration persistence after resizing the mirrored window;
- graceful reconnect after closing and reopening LonelyScreen.

Do not commit calibration files, screenshots containing unrelated desktop content, or Numba cache artifacts.

## Pull-request checklist

- Tests and Ruff pass.
- New behavior is documented.
- Direct dependencies remain pinned.
- Windows-only imports stay lazy so headless tests remain portable.
- Solver latency is checked with `scripts/benchmark_solver.py` when search behavior changes.
