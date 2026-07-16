# AGENTS.md

## Cursor Cloud specific instructions

### Project

Single Python package: **Block Blast Solver** (`block_blast_solver`). It is a
Windows desktop assistant that captures a mirrored Block Blast game, detects the
8x8 board + 3-piece inventory (OpenCV), ranks placement orders with a
Numba-accelerated bitboard solver, and draws a live OpenCV HUD. See `README.md`
and `CONTRIBUTING.md` for the full workflow.

### Environment

- Dependencies are installed into a virtualenv at `.venv/` by the startup update
  script (`pip install -e ".[dev]"`). Always invoke tools via `.venv/bin/...`
  (e.g. `.venv/bin/ruff`, `.venv/bin/coverage`, `.venv/bin/python`).
- `python3-venv` is a system package (needed for `python3 -m venv`); it is
  provisioned during environment setup, not by the update script.
- The `windows` extra (`dxcam`, `PyGetWindow`) is intentionally NOT installed:
  those packages are Windows-only and their imports are kept lazy so the rest of
  the package runs headless on Linux. Do not add them here.
- Runtime vision depends on `onnxruntime` and the committed weights under
  `block_blast_solver/models/`. PyTorch is only needed for the optional
  `train` extra when regenerating those weights. Skin fixtures live in
  `tests/fixtures/vision_skins/`. Set `VISION_FORCE_HEURISTIC=1` to force the
  classical detector path.

### Lint / test / build

Standard commands (from `README.md` / `CONTRIBUTING.md`), run with the venv:

- Lint: `.venv/bin/ruff check .`
- Test: `.venv/bin/coverage run -m unittest discover -s tests -v`
- Coverage (includes solver; Numba JIT is mostly untraceable):
  `.venv/bin/coverage report`
- Coverage gate (>=70% on non-solver modules):
  `.venv/bin/coverage report --omit=block_blast_solver/modules/solver.py --fail-under=70`
- No separate build step (editable install via `pip install -e`).

### Running the app

- The full app (`block-blast-solver` / `python -m block_blast_solver.main`)
  requires Windows: live DXGI capture (`dxcam`), a mirrored window titled
  `LonelyScreen AirPlay Receiver` (see `WINDOW_TITLE` in
  `block_blast_solver/config.py`), and an interactive OpenCV GUI. It cannot run
  end-to-end on the Linux cloud VM.
- On Linux, exercise the core pipeline headlessly instead: call
  `solver.solve_with_diagnostics(board, pieces)` and render the HUD via
  `Visualizer().draw_hud(...)` on a synthetic frame, then `cv2.imwrite` the
  result (no display needed). `scripts/benchmark_solver.py --iterations 10`
  measures warm solver latency.

### Gotchas

- The solver JIT-compiles on first call (Numba), so the first `solve*` call in a
  process is slow; latency benchmarks warm up once before timing.
- `config.BOARD_ROI` / `config.PIECES_ROI` default to `None`; set them (e.g. to
  `[x, y, w, h]` normalized ratios) before calling `draw_hud` if you want the
  board move overlays drawn on the game frame.
