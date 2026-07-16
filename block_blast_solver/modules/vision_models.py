from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from block_blast_solver import config

logger = logging.getLogger(__name__)

DEFAULT_BOARD_MODEL = config.BOARD_CELL_MODEL_PATH
DEFAULT_MASK_MODEL = config.INVENTORY_MASK_MODEL_PATH

_LOGGED_FAILURE_KEYS: set[str] = set()


def _log_model_failure_once(key: str, message: str, *args: object) -> None:
    if key in _LOGGED_FAILURE_KEYS:
        return
    _LOGGED_FAILURE_KEYS.add(key)
    logger.warning(message, *args)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    if float(np.min(array)) < 0.0 or float(np.max(array)) > 1.0:
        return _sigmoid(array).astype(np.float32)
    return array.astype(np.float32)


def _prepare_image_tensor(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected a BGR image with shape HxWx3")
    resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]


def _load_onnxruntime() -> Any | None:
    try:
        import onnxruntime as ort
    except Exception as error:  # pragma: no cover - exercised only when dependency import fails.
        _log_model_failure_once("onnxruntime-import", "ONNX runtime unavailable; using heuristics instead: %s", error)
        return None
    return ort


def _load_session(model_path: str, label: str) -> Any | None:
    path = Path(model_path)
    if not path.is_file():
        _log_model_failure_once(f"missing:{path}", "Missing %s weights at %s; using heuristics instead", label, path)
        return None

    ort = _load_onnxruntime()
    if ort is None:
        return None

    try:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:  # pragma: no cover - depends on invalid or incompatible model files.
        _log_model_failure_once(f"session:{path}", "Failed to load %s weights at %s: %s", label, path, error)
        return None


def board_probs_are_occluded(probs: np.ndarray) -> bool:
    normalized = _normalize_probabilities(probs)
    uncertain_count = int(np.sum(np.abs(normalized - 0.5) < 0.15))
    return uncertain_count > 16


class BoardCellClassifier:
    def __init__(self, session: Any):
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def predict_proba(self, crop_bgr: np.ndarray) -> float:
        tensor = _prepare_image_tensor(crop_bgr, (32, 32))
        output = self._session.run(None, {self._input_name: tensor})[0]
        probs = _normalize_probabilities(np.asarray(output, dtype=np.float32).reshape(-1))
        return float(probs[0])


class InventorySlotMasker:
    def __init__(self, session: Any):
        self._session = session
        self._input_name = session.get_inputs()[0].name

    def predict_mask(self, slot_bgr: np.ndarray) -> np.ndarray:
        original_height, original_width = slot_bgr.shape[:2]
        tensor = _prepare_image_tensor(slot_bgr, (128, 128))
        output = self._session.run(None, {self._input_name: tensor})[0]
        mask = np.squeeze(np.asarray(output, dtype=np.float32))
        if mask.ndim != 2:
            raise ValueError("inventory mask model must return a single 2D mask")
        normalized_mask = _normalize_probabilities(mask)
        resized_mask = cv2.resize(normalized_mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        return resized_mask.astype(np.float32)


class ModelRegistry:
    _instance: ClassVar["ModelRegistry" | None] = None

    def __init__(
        self,
        board_classifier: BoardCellClassifier | None,
        inventory_masker: InventorySlotMasker | None,
    ):
        self.board_classifier = board_classifier
        self.inventory_masker = inventory_masker
        self.using_learned = board_classifier is not None and inventory_masker is not None

    @classmethod
    def get(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls._build()
        return cls._instance

    @classmethod
    def _build(cls) -> "ModelRegistry":
        if config.vision_force_heuristic():
            return cls(None, None)

        board_session = _load_session(DEFAULT_BOARD_MODEL, "board cell classifier")
        mask_session = _load_session(DEFAULT_MASK_MODEL, "inventory slot masker")
        if board_session is None or mask_session is None:
            return cls(None, None)
        return cls(BoardCellClassifier(board_session), InventorySlotMasker(mask_session))

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
