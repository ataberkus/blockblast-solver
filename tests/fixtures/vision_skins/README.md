## Vision skin fixtures

This directory stores synthetic stand-in screenshots for the learned block
detection fixture pack from the 2026-07-16 design brainstorm. The original
brainstorm screenshots are not available as binary assets in this environment,
so these PNGs recreate the intended variety instead: gem, cake/wafer,
watermelon, wood, pearlescent mixed-color, heart pastel, and stone-like skins.

Each fixture uses a matching `<name>.json` label file with this schema:

- `board_roi`: normalized `[x, y, w, h]` for the full 8x8 board
- `pieces_roi`: normalized `[x, y, w, h]` for the full three-slot inventory strip
- `board`: 8x8 matrix of `0` and `1` occupancy values
- `pieces`: list of three item values, each either `null` or a 2D `0`/`1`
  matrix describing that inventory piece shape

The stand-ins intentionally vary color, texture, bevels, drop shadows, and
mixed-color cells so later learned-vision tasks can regress against a broader
set of block appearances without changing the runtime `vision.py` heuristics in
this task.
