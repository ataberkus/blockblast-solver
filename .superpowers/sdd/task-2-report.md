# Task 2 report: Config knobs + ModelRegistry with heuristic fallback

## Scope

Implemented only Task 2 from the approved learned block detection plan:

- added learned-vision config knobs in `block_blast_solver/config.py`
- added `block_blast_solver/modules/vision_models.py`
- added empty packaged model namespace at `block_blast_solver/models/__init__.py`
- added `tests/test_vision_models.py`
- updated packaging metadata in `pyproject.toml`
- mirrored `onnxruntime==1.27.1` into `requirements.txt`

I did not wire learned models into `get_board_state()` or `get_pieces()`, and I
did not add real `.onnx` weight files.

## TDD evidence

### RED: new registry/config tests before implementation

Command:

` .venv/bin/python -m unittest tests.test_vision_models -v `

Output:

```text
test_vision_models (unittest.loader._FailedTest.test_vision_models) ... ERROR

======================================================================
ERROR: test_vision_models (unittest.loader._FailedTest.test_vision_models)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_vision_models
...
ImportError: cannot import name 'vision_models' from 'block_blast_solver.modules'

----------------------------------------------------------------------
Ran 1 test in 0.001s

FAILED (errors=1)
```

Result: expected RED. The new test failed because the `vision_models` module did
not exist yet.

### GREEN: focused config/registry suite after implementation

Command:

` .venv/bin/python -m unittest tests.test_vision_models -v `

Output:

```text
test_exports_default_model_paths_and_thresholds ... ok
test_reads_model_path_and_force_heuristic_overrides_from_env ... ok
test_force_heuristic_disables_models ... ok
test_missing_weights_fall_back ... ok
test_occlusion_helper_flags_uncertain_board ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.003s

OK
```

Result: expected GREEN. The new config knobs, heuristic-force gate, missing
weight fallback, and occlusion helper all passed.

## Dependency installation note

The task brief pinned `onnxruntime==1.27.1` and required an editable reinstall.
I kept that exact pin in `pyproject.toml` and `requirements.txt`, then ran the
required install command.

### Editable reinstall attempt

Command:

` .venv/bin/python -m pip install -e ".[dev]" `

Output summary:

```text
ERROR: Could not find a version that satisfies the requirement onnxruntime==1.27.1
ERROR: No matching distribution found for onnxruntime==1.27.1
```

### Root-cause check

Command:

` .venv/bin/python -m pip index versions onnxruntime `

Output:

```text
onnxruntime (1.27.0)
Available versions: 1.27.0, 1.26.0, ...
```

Result: the configured index available to this environment does not publish
`1.27.1`; it stops at `1.27.0`. Because `vision_models.py` loads ONNX lazily and
Task 2 intentionally falls back when weights are absent, the source tree still
tests cleanly without that reinstall succeeding.

## Verification

### Lint

Command:

` .venv/bin/ruff check . `

Output:

```text
All checks passed!
```

### Full suite

Command:

` .venv/bin/coverage run -m unittest discover -s tests -v `

Output summary:

```text
Ran 67 tests in 6.459s

OK
```

### Coverage reports

Command:

` .venv/bin/coverage report `

Output summary:

```text
TOTAL                                            1378    597    516     53    53%
```

Command:

` .venv/bin/coverage report --omit=block_blast_solver/modules/solver.py --fail-under=70 `

Output summary:

```text
TOTAL                                             745     85    250     48    86%
```

## Self-review

- `ModelRegistry` is fully lazy and stays isolated from `vision.py`, which keeps
  Task 2 inside scope.
- Missing or unloadable model paths degrade to `using_learned=False` with no
  partial learned-mode state.
- `requirements-dev.txt` did not need a direct edit because it already delegates
  to `requirements.txt`.
- The new `block_blast_solver/models/` package is present for future `.onnx`
  assets, but no weights are committed in this task.

## Concerns

- The exact `onnxruntime==1.27.1` pin from the brief is not installable from the
  package index exposed in this environment. The pin remains declared verbatim,
  but dependency installation is blocked until that artifact becomes available or
  the brief is corrected.
