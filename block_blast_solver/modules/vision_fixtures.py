from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

DEFAULT_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "vision_skins"


@dataclass(frozen=True)
class SkinFixture:
    name: str
    image: np.ndarray
    board: np.ndarray
    pieces: List[Optional[np.ndarray]]
    board_roi: list[float]
    pieces_roi: list[float]
    label_path: Path
    image_path: Path


def validate_skin_label(data: object) -> None:
    if not isinstance(data, dict):
        raise ValueError("label root must be an object")
    for key in ("board_roi", "pieces_roi", "board", "pieces"):
        if key not in data:
            raise ValueError(f"missing key: {key}")
    for roi_key in ("board_roi", "pieces_roi"):
        roi = data[roi_key]
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError(f"{roi_key} must be 4 floats")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in roi):
            raise ValueError(f"{roi_key} values must be numbers")
    board = data["board"]
    if not isinstance(board, list) or len(board) != 8:
        raise ValueError("board must be 8 rows")
    for row in board:
        if not isinstance(row, list) or len(row) != 8:
            raise ValueError("board rows must have 8 cells")
        if any(cell not in (0, 1) for cell in row):
            raise ValueError("board cells must be 0 or 1")
    pieces = data["pieces"]
    if not isinstance(pieces, list) or len(pieces) != 3:
        raise ValueError("pieces must be a list of length 3")
    for piece in pieces:
        if piece is None:
            continue
        if not isinstance(piece, list) or not piece:
            raise ValueError("piece matrix must be a non-empty 2D list")
        width = len(piece[0])
        if width < 1:
            raise ValueError("piece matrix has empty row")
        for row in piece:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError("piece rows must share width")
            if any(cell not in (0, 1) for cell in row):
                raise ValueError("piece cells must be 0 or 1")


def load_skin_fixture(label_path: Path) -> SkinFixture:
    label_path = Path(label_path)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    validate_skin_label(data)
    image_path = label_path.with_suffix(".png")
    if not image_path.exists():
        image_path = label_path.with_suffix(".jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"no image beside {label_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    pieces: List[Optional[np.ndarray]] = []
    for piece in data["pieces"]:
        if piece is None:
            pieces.append(None)
        else:
            pieces.append(np.asarray(piece, dtype=np.uint8))
    return SkinFixture(
        name=label_path.stem,
        image=image,
        board=np.asarray(data["board"], dtype=np.uint8),
        pieces=pieces,
        board_roi=[float(value) for value in data["board_roi"]],
        pieces_roi=[float(value) for value in data["pieces_roi"]],
        label_path=label_path,
        image_path=image_path,
    )


def iter_skin_fixtures(root: Path | None = None) -> Iterator[SkinFixture]:
    fixture_root = Path(root) if root is not None else DEFAULT_FIXTURE_ROOT
    for label_path in sorted(fixture_root.glob("*.json")):
        yield load_skin_fixture(label_path)
