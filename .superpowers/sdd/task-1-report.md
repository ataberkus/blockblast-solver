# Task 1 report: Skin fixture schema, loader, and labeled screenshots

## Scope

Implemented only Task 1 from the approved learned block detection plan:

- added `block_blast_solver/modules/vision_fixtures.py`
- added `tests/test_vision_fixture_schema.py`
- added synthetic stand-in fixture PNG/JSON pairs under `tests/fixtures/vision_skins/`
- added `tests/fixtures/vision_skins/README.md`

I did not change `vision.py` detection logic.

## TDD evidence

### RED 1: missing loader module

Command:

` .venv/bin/python -m unittest tests.test_vision_fixture_schema -v `

Output:

```text
test_vision_fixture_schema (unittest.loader._FailedTest.test_vision_fixture_schema) ... ERROR

======================================================================
ERROR: test_vision_fixture_schema (unittest.loader._FailedTest.test_vision_fixture_schema)
----------------------------------------------------------------------
ImportError: Failed to import test module: test_vision_fixture_schema
Traceback (most recent call last):
  File "/usr/lib/python3.12/unittest/loader.py", line 137, in loadTestsFromName
    module = __import__(module_name)
             ^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/tests/test_vision_fixture_schema.py", line 8, in <module>
    from block_blast_solver.modules.vision_fixtures import load_skin_fixture, validate_skin_label
ModuleNotFoundError: No module named 'block_blast_solver.modules.vision_fixtures'

----------------------------------------------------------------------
Ran 1 test in 0.000s

FAILED (errors=1)
```

Result: expected RED. The new test failed because the module did not exist yet.

### GREEN 1: loader + schema implementation

Command:

` .venv/bin/python -m unittest tests.test_vision_fixture_schema -v `

Output:

```text
test_load_skin_fixture_round_trip (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_load_skin_fixture_round_trip) ... ok
test_validate_accepts_minimal_label (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_validate_accepts_minimal_label) ... ok
test_validate_rejects_wrong_board_shape (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_validate_rejects_wrong_board_shape) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.021s

OK
```

Result: expected GREEN after adding `vision_fixtures.py`.

### RED 2: repository fixture pack absent

Command:

` .venv/bin/python -m unittest tests.test_vision_fixture_schema -v `

Output:

```text
test_iter_skin_fixtures_loads_expected_repo_fixture_names (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_iter_skin_fixtures_loads_expected_repo_fixture_names) ... FAIL
test_load_skin_fixture_round_trip (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_load_skin_fixture_round_trip) ... ok
test_validate_accepts_minimal_label (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_validate_accepts_minimal_label) ... ok
test_validate_rejects_wrong_board_shape (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_validate_rejects_wrong_board_shape) ... ok

======================================================================
FAIL: test_iter_skin_fixtures_loads_expected_repo_fixture_names (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_iter_skin_fixtures_loads_expected_repo_fixture_names)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/workspace/tests/test_vision_fixture_schema.py", line 17, in test_iter_skin_fixtures_loads_expected_repo_fixture_names
    self.assertEqual(
AssertionError: Lists differ: [] != ['cake_wafer', 'gem_busy', 'gem_five_line'[116 chars]sic']
```

Result: expected RED. The fixture directory had no JSON/PNG pack yet.

### GREEN 2: synthetic labeled fixtures added

Command:

` .venv/bin/python -m unittest tests.test_vision_fixture_schema -v `

Output:

```text
test_iter_skin_fixtures_loads_expected_repo_fixture_names (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_iter_skin_fixtures_loads_expected_repo_fixture_names) ... ok
test_load_skin_fixture_round_trip (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_load_skin_fixture_round_trip) ... ok
test_validate_accepts_minimal_label (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_validate_accepts_minimal_label) ... ok
test_validate_rejects_wrong_board_shape (tests.test_vision_fixture_schema.VisionFixtureSchemaTests.test_validate_rejects_wrong_board_shape) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.058s

OK
```

Result: expected GREEN after generating the synthetic stand-in skins and labels.

## Fixture pack notes

Because the brainstorm screenshots were not available as binary assets in this
environment, I followed `task-1-fixture-note.md` and created synthetic OpenCV
stand-ins instead.

Committed fixture names:

- `cake_wafer`
- `gem_busy`
- `gem_five_line`
- `gem_two_rows`
- `heart_pastel`
- `pearl_mixed`
- `stone_corner`
- `watermelon_green`
- `watermelon_purple`
- `wood_classic`

The pack intentionally varies:

- flat and faceted gem blocks
- two-tone cake/wafer blocks
- watermelon rind + seed details
- wood grain noise
- stone texture noise
- pearlescent gradients
- mixed-color cells within one piece
- inventory drop shadows

## Verification

### Smoke-load command

Command:

` .venv/bin/python -c "from block_blast_solver.modules.vision_fixtures import iter_skin_fixtures; print([f.name for f in iter_skin_fixtures()])" `

Output:

```text
['cake_wafer', 'gem_busy', 'gem_five_line', 'gem_two_rows', 'heart_pastel', 'pearl_mixed', 'stone_corner', 'watermelon_green', 'watermelon_purple', 'wood_classic']
```

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
Ran 60 tests in 15.309s

OK
```

### Coverage reports

Command:

` .venv/bin/coverage report `

Output summary:

```text
TOTAL                                            1271    563    496     49    53%
```

Command:

` .venv/bin/coverage report --omit=block_blast_solver/modules/solver.py --fail-under=70 `

Output summary:

```text
TOTAL                                             638     51    230     44    89%
```

## Self-review

- `vision_fixtures.py` stays test-shared and importable without touching runtime
  detection paths.
- The loader accepts `.png` first and falls back to `.jpg`, matching the design
  fixture requirement.
- The fixture test now proves both schema validation and that the repo contains
  the expected named pack.
- The synthetic fixture set includes 10 distinct skins, which exceeds the
  minimum note requirement of 8.

## Concerns

None. The one notable constraint is intentional and documented: the fixture
images are synthetic stand-ins because the brainstorm screenshots were not
available as binary files.
