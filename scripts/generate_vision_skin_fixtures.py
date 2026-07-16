#!/usr/bin/env python3
"""Regenerate synthetic skin fixtures with clear empty/occupied contrast."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from block_blast_solver import config

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "vision_skins"
FRAME_W, FRAME_H = 480, 720
BOARD_ROI = [0.125, 0.12, 0.75, 0.50]
PIECES_ROI = [0.10, 0.72, 0.80, 0.20]


SKINS = {
    "pearl_mixed": {
        "bg": (195, 140, 210),
        "empty": (70, 40, 80),
        "palette": [(220, 180, 255), (240, 200, 120), (200, 230, 255)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
        ],
        "pieces": [[[1, 0], [1, 0], [1, 1]], [[1, 1], [1, 1]], [[1, 1, 0], [0, 1, 1]]],
    },
    "gem_busy": {
        "bg": (120, 60, 90),
        "empty": (35, 25, 40),
        "palette": [(40, 160, 255), (60, 220, 80), (40, 80, 220), (0, 200, 255), (200, 80, 200)],
        "board": [
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ],
        "pieces": [[[1, 1, 1], [1, 1, 1]], [[0, 0, 1], [1, 1, 1]], [[1], [1], [1], [1]]],
    },
    "gem_five_line": {
        "bg": (110, 50, 90),
        "empty": (30, 20, 35),
        "palette": [(255, 200, 80), (80, 220, 255), (200, 80, 200), (60, 90, 220), (40, 180, 255)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1],
        ],
        "pieces": [[[1, 1, 1, 1, 1]], None, None],
    },
    "cake_wafer": {
        "bg": (70, 90, 130),
        "empty": (40, 50, 70),
        "palette": [(180, 200, 255), (140, 180, 220), (120, 160, 200)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1],
        ],
        "pieces": [[[1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]],
    },
    "gem_two_rows": {
        "bg": (100, 50, 80),
        "empty": (25, 20, 30),
        "palette": [(80, 220, 80), (255, 200, 60), (220, 80, 80), (200, 100, 255), (60, 90, 220)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1],
        ],
        "pieces": [[[1], [1], [1], [1]], [[1, 1, 1], [1, 1, 1]], [[1, 1], [1, 1]]],
    },
    "watermelon_purple": {
        "bg": (90, 40, 90),
        "empty": (30, 20, 35),
        "palette": [(60, 60, 220), (80, 180, 60)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "pieces": [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1], [1], [1], [1], [1]],
        ],
    },
    "heart_pastel": {
        "bg": (90, 100, 140),
        "empty": (45, 50, 65),
        "palette": [(180, 160, 255), (160, 200, 230)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
        "pieces": [
            [[1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1]],
        ],
    },
    "watermelon_green": {
        "bg": (70, 140, 70),
        "empty": (30, 60, 35),
        "palette": [(60, 60, 220), (50, 180, 50)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "pieces": [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1], [1], [1], [1], [1]],
        ],
    },
    "wood_classic": {
        "bg": (160, 190, 220),
        "empty": (50, 70, 90),
        "palette": [(40, 140, 255), (60, 180, 60), (200, 120, 40), (180, 80, 40)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
        ],
        "pieces": [[[1, 1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1], [1, 1]]],
    },
    "stone_corner": {
        "bg": (90, 100, 110),
        "empty": (45, 50, 55),
        "palette": [(180, 200, 210), (160, 175, 185)],
        "board": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ],
        "pieces": [[[1, 1, 1, 1, 1]], [[1, 1, 1, 1, 1]], [[1, 1, 1], [0, 0, 1], [0, 0, 1]]],
    },
}


def _draw_block(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int], style: str) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    highlight = tuple(min(255, c + 40) for c in color)
    shadow = tuple(max(0, c - 50) for c in color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), highlight, 1)
    cv2.line(frame, (x1, y2 - 2), (x2, y2 - 2), shadow, 2)
    if style == "gem":
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), max(2, (x2 - x1) // 6), highlight, -1)
    elif style == "cake":
        mid = (y1 + y2) // 2
        cv2.line(frame, (x1 + 2, mid), (x2 - 2, mid), (255, 255, 255), 1)
        for px in range(x1 + 4, x2 - 3, 6):
            cv2.circle(frame, (px, mid - 3), 1, (0, 0, 255), -1)
    elif style == "watermelon":
        cv2.line(frame, (x1 + 1, y2 - 4), (x2 - 1, y2 - 4), (60, 200, 60), 2)
        cv2.circle(frame, ((x1 + x2) // 2 - 3, (y1 + y2) // 2), 1, (0, 0, 0), -1)
        cv2.circle(frame, ((x1 + x2) // 2 + 3, (y1 + y2) // 2 + 2), 1, (0, 0, 0), -1)
    elif style == "wood":
        for yy in range(y1 + 3, y2 - 2, 4):
            cv2.line(frame, (x1 + 2, yy), (x2 - 2, yy), shadow, 1)


def _style_for(name: str) -> str:
    if "cake" in name:
        return "cake"
    if "watermelon" in name:
        return "watermelon"
    if "wood" in name:
        return "wood"
    if "stone" in name or "heart" in name or "pearl" in name:
        return "gem"
    return "gem"


def _roi_pixels(roi: list[float]) -> tuple[int, int, int, int]:
    x = int(roi[0] * FRAME_W)
    y = int(roi[1] * FRAME_H)
    w = int(roi[2] * FRAME_W)
    h = int(roi[3] * FRAME_H)
    return x, y, w, h


def _draw_board(frame: np.ndarray, board: list[list[int]], empty: tuple[int, int, int], palette: list[tuple[int, int, int]], style: str) -> None:
    bx, by, bw, bh = _roi_pixels(BOARD_ROI)
    cell_w = bw / 8.0
    cell_h = bh / 8.0
    for row in range(8):
        for col in range(8):
            x1 = bx + int(col * cell_w)
            y1 = by + int(row * cell_h)
            x2 = bx + int((col + 1) * cell_w) - 1
            y2 = by + int((row + 1) * cell_h) - 1
            if board[row][col] == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), empty, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), tuple(min(255, c + 25) for c in empty), 1)
            else:
                color = palette[(row * 8 + col) % len(palette)]
                _draw_block(frame, x1 + 1, y1 + 1, x2 - 1, y2 - 1, color, style)


def _draw_pieces(frame: np.ndarray, pieces: list, palette: list[tuple[int, int, int]], style: str) -> None:
    px, py, pw, ph = _roi_pixels(PIECES_ROI)
    bx, by, bw, bh = _roi_pixels(BOARD_ROI)
    inv_cell = (bw / 8.0) * config.PIECE_SCALE_FACTOR
    slot_w = pw / 3.0
    for slot, piece in enumerate(pieces):
        slot_x1 = px + int(slot * slot_w)
        slot_x2 = px + int((slot + 1) * slot_w)
        if piece is None:
            continue
        rows = len(piece)
        cols = len(piece[0])
        block = max(8, int(round(inv_cell)))
        piece_w = cols * block
        piece_h = rows * block
        start_x = slot_x1 + max(0, ((slot_x2 - slot_x1) - piece_w) // 2)
        start_y = py + max(0, (ph - piece_h) // 2)
        # drop shadow
        cv2.rectangle(
            frame,
            (start_x + 4, start_y + 6),
            (start_x + piece_w + 2, start_y + piece_h + 4),
            (20, 20, 20),
            -1,
        )
        color = palette[slot % len(palette)]
        for r in range(rows):
            for c in range(cols):
                if piece[r][c] != 1:
                    continue
                x1 = start_x + c * block
                y1 = start_y + r * block
                # mixed-color variant for pearl/heart
                cell_color = palette[(slot + r + c) % len(palette)] if style in {"gem", "cake"} else color
                _draw_block(frame, x1, y1, x1 + block - 2, y1 + block - 2, cell_color, style)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, spec in SKINS.items():
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        frame[:] = spec["bg"]
        style = _style_for(name)
        _draw_board(frame, spec["board"], spec["empty"], spec["palette"], style)
        _draw_pieces(frame, spec["pieces"], spec["palette"], style)
        cv2.imwrite(str(ROOT / f"{name}.png"), frame)
        label = {
            "board_roi": BOARD_ROI,
            "pieces_roi": PIECES_ROI,
            "board": spec["board"],
            "pieces": spec["pieces"],
        }
        (ROOT / f"{name}.json").write_text(json.dumps(label, indent=2) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
