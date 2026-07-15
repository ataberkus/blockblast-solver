"""State management for stable vision detections and solver results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def copy_pieces(pieces: List[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    return [piece.copy() if piece is not None else None for piece in pieces]


def state_changed(
    board1: np.ndarray,
    board2: np.ndarray,
    pieces1: List[Optional[np.ndarray]],
    pieces2: List[Optional[np.ndarray]],
) -> bool:
    if not np.array_equal(board1, board2) or len(pieces1) != len(pieces2):
        return True
    return any(
        (piece1 is None) != (piece2 is None)
        or (piece1 is not None and not np.array_equal(piece1, piece2))
        for piece1, piece2 in zip(pieces1, pieces2)
    )


@dataclass
class DetectionUpdate:
    board: np.ndarray
    pieces: List[Optional[np.ndarray]]
    changed: bool


@dataclass
class AppState:
    last_board: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=np.uint8))
    last_pieces: List[Optional[np.ndarray]] = field(default_factory=lambda: [None, None, None])
    last_moves: Optional[List[Dict[str, Any]]] = None
    last_score: float = 0.0
    last_diagnostics: Optional[Dict[str, Any]] = None
    pending_board: Optional[np.ndarray] = None
    pending_pieces: List[Optional[np.ndarray]] = field(default_factory=lambda: [None, None, None])
    stable_frame_count: int = 0

    def reset(self) -> None:
        self.last_board = np.zeros((8, 8), dtype=np.uint8)
        self.last_pieces = [None, None, None]
        self.last_moves = None
        self.last_score = 0.0
        self.last_diagnostics = None
        self.pending_board = None
        self.pending_pieces = [None, None, None]
        self.stable_frame_count = 0

    def observe(
        self,
        board: np.ndarray,
        pieces: List[Optional[np.ndarray]],
        required_stable_frames: int,
        occluded: bool = False,
    ) -> Optional[DetectionUpdate]:
        if occluded:
            self.stable_frame_count = 0
            return None

        if self.pending_board is not None and not state_changed(
            board,
            self.pending_board,
            pieces,
            self.pending_pieces,
        ):
            self.stable_frame_count += 1
        else:
            self.pending_board = board.copy()
            self.pending_pieces = copy_pieces(pieces)
            self.stable_frame_count = 1

        if self.stable_frame_count < max(1, required_stable_frames):
            return None

        stable_board = self.pending_board.copy()
        stable_pieces = copy_pieces(self.pending_pieces)
        return DetectionUpdate(
            board=stable_board,
            pieces=stable_pieces,
            changed=state_changed(stable_board, self.last_board, stable_pieces, self.last_pieces),
        )

    def accept(
        self,
        board: np.ndarray,
        pieces: List[Optional[np.ndarray]],
        moves: Optional[List[Dict[str, Any]]],
        score: float,
        diagnostics: Optional[Dict[str, Any]],
    ) -> None:
        self.last_board = board.copy()
        self.last_pieces = copy_pieces(pieces)
        self.last_moves = moves
        self.last_score = score
        self.last_diagnostics = diagnostics
