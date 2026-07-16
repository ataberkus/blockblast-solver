## Vision skin fixtures

Synthetic stand-in screenshots for learned block detection. Original brainstorm
screenshots were unavailable as binaries, so these PNGs recreate gem, cake/wafer,
watermelon, wood, pearlescent mixed-color, heart pastel, and stone-like skins with
clear empty-vs-occupied contrast.

Regenerate with:

```bash
.venv/bin/python scripts/generate_vision_skin_fixtures.py
```

Each fixture uses a matching `<name>.json` label file:

- `board_roi`: normalized `[x, y, w, h]` for the full 8x8 board
- `pieces_roi`: normalized `[x, y, w, h]` for the full three-slot inventory strip
- `board`: 8x8 matrix of `0` and `1` occupancy values
- `pieces`: list of three values, each either `null` or a 2D `0`/`1` matrix

Committed ONNX weights under `block_blast_solver/models/` are trained from these
fixtures via `scripts/export_vision_training_data.py` and
`scripts/train_vision_models.py`.
