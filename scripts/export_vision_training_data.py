#!/usr/bin/env python3
"""Export board-cell crops and inventory slot masks from skin fixtures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from block_blast_solver import config
from block_blast_solver.modules.vision_fixtures import SkinFixture, iter_skin_fixtures


def _roi_pixels(frame: np.ndarray, roi: list[float]) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x = max(0, int(roi[0] * width))
    y = max(0, int(roi[1] * height))
    w = min(width - x, int(roi[2] * width))
    h = min(height - y, int(roi[3] * height))
    return x, y, w, h


def _export_board_cells(fixture: SkinFixture, occupied_dir: Path, empty_dir: Path) -> tuple[int, int]:
    bx, by, bw, bh = _roi_pixels(fixture.image, fixture.board_roi)
    board_crop = fixture.image[by:by + bh, bx:bx + bw]
    cell_w = bw / 8.0
    cell_h = bh / 8.0
    occupied = 0
    empty = 0

    for row in range(8):
        for col in range(8):
            c_x1 = int(col * cell_w)
            c_y1 = int(row * cell_h)
            c_w = max(1, int(cell_w))
            c_h = max(1, int(cell_h))
            offset_x = int(c_w * 0.20)
            offset_y = int(c_h * 0.20)
            inner_w = max(1, int(c_w * 0.60))
            inner_h = max(1, int(c_h * 0.60))
            crop = board_crop[c_y1 + offset_y:c_y1 + offset_y + inner_h, c_x1 + offset_x:c_x1 + offset_x + inner_w]
            if crop.size == 0:
                continue
            label = int(fixture.board[row, col])
            target_dir = occupied_dir if label == 1 else empty_dir
            filename = f"{fixture.name}_r{row}_c{col}.png"
            cv2.imwrite(str(target_dir / filename), crop)
            if label == 1:
                occupied += 1
            else:
                empty += 1
    return occupied, empty


def _rasterize_piece_mask(
    slot_shape: tuple[int, int],
    piece: np.ndarray,
    inv_cell_w: float,
    inv_cell_h: float,
    slot_crop: np.ndarray | None = None,
) -> np.ndarray:
    del slot_crop  # labels are rasterized on the same centered grid used by fixture generation
    slot_h, slot_w = slot_shape
    mask = np.zeros((slot_h, slot_w), dtype=np.uint8)
    rows, cols = piece.shape
    block_w = max(1, int(round(inv_cell_w)))
    block_h = max(1, int(round(inv_cell_h)))
    piece_w = cols * block_w
    piece_h = rows * block_h
    start_x = max(0, (slot_w - piece_w) // 2)
    start_y = max(0, (slot_h - piece_h) // 2)

    for r in range(rows):
        for c in range(cols):
            if piece[r, c] != 1:
                continue
            x1 = start_x + c * block_w
            y1 = start_y + r * block_h
            x2 = min(slot_w, x1 + block_w)
            y2 = min(slot_h, y1 + block_h)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255
    return mask


def _export_inventory(fixture: SkinFixture, images_dir: Path, masks_dir: Path) -> int:
    px, py, pw, ph = _roi_pixels(fixture.image, fixture.pieces_roi)
    pieces_crop = fixture.image[py:py + ph, px:px + pw]
    bx, by, bw, bh = _roi_pixels(fixture.image, fixture.board_roi)
    board_cell_w = bw / 8.0
    board_cell_h = bh / 8.0
    scale = getattr(config, "PIECE_SCALE_FACTOR", 0.47)
    inv_cell_w = board_cell_w * scale
    inv_cell_h = board_cell_h * scale
    slot_w = pw / 3.0
    exported = 0

    for slot_index, piece in enumerate(fixture.pieces):
        slot_x1 = int(slot_index * slot_w)
        slot_x2 = int((slot_index + 1) * slot_w)
        slot_crop = pieces_crop[0:ph, slot_x1:slot_x2]
        if slot_crop.size == 0:
            continue
        if piece is None:
            mask = np.zeros(slot_crop.shape[:2], dtype=np.uint8)
        else:
            mask = _rasterize_piece_mask(slot_crop.shape[:2], piece, inv_cell_w, inv_cell_h, slot_crop)
        stem = f"{fixture.name}_slot{slot_index}"
        cv2.imwrite(str(images_dir / f"{stem}.png"), slot_crop)
        cv2.imwrite(str(masks_dir / f"{stem}.png"), mask)
        exported += 1
    return exported


def export_training_data(out_dir: Path, fixture_root: Path | None = None) -> dict:
    fixtures = list(iter_skin_fixtures(fixture_root))
    if not fixtures:
        raise SystemExit("No skin fixtures found; refusing to export an empty dataset")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    occupied_dir = out_dir / "board_cells" / "occupied"
    empty_dir = out_dir / "board_cells" / "empty"
    images_dir = out_dir / "inventory" / "images"
    masks_dir = out_dir / "inventory" / "masks"
    for path in (occupied_dir, empty_dir, images_dir, masks_dir):
        path.mkdir(parents=True, exist_ok=True)

    occupied_total = 0
    empty_total = 0
    inventory_total = 0
    for fixture in fixtures:
        occupied, empty = _export_board_cells(fixture, occupied_dir, empty_dir)
        inventory = _export_inventory(fixture, images_dir, masks_dir)
        occupied_total += occupied
        empty_total += empty
        inventory_total += inventory

    manifest = {
        "fixtures": [fixture.name for fixture in fixtures],
        "board_cells_occupied": occupied_total,
        "board_cells_empty": empty_total,
        "inventory_pairs": inventory_total,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=None,
        help="Optional override for skin fixture root",
    )
    args = parser.parse_args()
    manifest = export_training_data(args.out, args.fixture_root)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
