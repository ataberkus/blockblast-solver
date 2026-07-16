import json
import logging
import math
import os
from pathlib import Path
from typing import Sequence, Tuple

logger = logging.getLogger(__name__)

# =====================================================================
# BLOCK BLAST SOLVER - KONFİGÜRASYON VE PAYLAŞILAN DURUM MODÜLÜ (config.py)
# =====================================================================

# Pencere başlığı ve kalibrasyon dosya yolu tanımları
WINDOW_TITLE = "LonelyScreen AirPlay Receiver"


def _default_calibration_file() -> Path:
    override = os.environ.get("BLOCK_BLAST_CALIBRATION_FILE")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        config_root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        config_root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_root / "block-blast-solver" / "calibration_data.json"


CALIBRATION_FILE = str(_default_calibration_file())

_MODELS_DIR = Path(__file__).resolve().parent / "models"
BOARD_CELL_MODEL_PATH = os.environ.get(
    "BLOCK_BLAST_BOARD_CELL_MODEL",
    str(_MODELS_DIR / "board_cell_classifier.onnx"),
)
INVENTORY_MASK_MODEL_PATH = os.environ.get(
    "BLOCK_BLAST_INVENTORY_MASK_MODEL",
    str(_MODELS_DIR / "inventory_slot_masker.onnx"),
)
T_BOARD = 0.5
T_MASK = 0.5
T_CELL = 0.30


def vision_force_heuristic() -> bool:
    return os.environ.get("VISION_FORCE_HEURISTIC", "").strip().lower() in {"1", "true", "yes", "on"}


VISION_FORCE_HEURISTIC = vision_force_heuristic()

# Envanter parçalarının tahta hücrelerine göre ölçek çarpanı
PIECE_SCALE_FACTOR = 0.47

# Yeni algılanan tahta/parça durumu solver'a verilmeden önce kaç ardışık kare aynı kalmalı
DETECTION_STABLE_FRAMES = 6

# Helezonik/Kombinatoryal Yapay Zeka Ağırlıkları (Heuristic AI Weights)
# En iyi hamle sırasını ve koordinatını belirlemek için tahta skorunu hesaplar.
W_CLEAR = 500.0      # Satır/Sütun temizleme ödülü
W_EMPTY = 15.0       # Boş hücre bırakma ödülü
W_HOLES = -80.0      # Kapatılmış, doldurulması imkansız tekli boşluklar için ağır ceza puanı
W_BUMPINESS = -10.0  # Yan yana sütunlar arasındaki yükseklik farkı (engebelilik) cezası
W_READINESS = 350.0  # Büyük blokların (3x3, 5x1, 1x5 vb.) sığabileceği açık alanları tutma ödülü

# Survival-focused evaluator weights
W_FUTURE_FITS = 42.0
W_LARGEST_REGION = 18.0
W_SMALL_REGION_PENALTY = -35.0
W_LINE_READINESS_SURVIVAL = 28.0
W_TRAP_PENALTY = -55.0

# Monte Carlo next-piece survival weights and limits
MONTE_CARLO_NORMAL_SAMPLES = 2
MONTE_CARLO_DANGER_SAMPLES = 4
MONTE_CARLO_DANGER_FILLED_CELLS = 38
MONTE_CARLO_MIN_FUTURE_FITS = 8
MONTE_CARLO_SEED = 0xC0FFEE
W_MONTE_CARLO_SURVIVAL = 120.0
W_MONTE_CARLO_CLEAR_ROUTES = 18.0
W_MONTE_CARLO_FUTURE_FITS = 12.0

# Deterministic recursion budget. Zero enables exhaustive search.
# The default stays below the 250 ms warm-latency target on the benchmark fixture.
SEARCH_NODE_BUDGET = 2500

# Streak / combo continuity weights (Section 1 + Section 2 of streak spec)
W_STREAK_CONTINUE        =  250.0   # per placement in current plan that clears >= 1 line
W_STREAK_BREAK_PENALTY   = -900.0   # per placement in current plan that clears 0 lines
W_STREAK_PERFECT_BONUS   =  600.0   # one-shot bonus when all 3 placements clear
W_MONTE_CARLO_STREAK     =  200.0   # scaled by avg_streak_continuity in [0.0, 1.0]

# Kalibrasyon ROI (Region of Interest) Değişkenleri
# Pencere boyutuna oranlanmış şekilde (normalized ratios: 0.0 - 1.0) saklanır.
BOARD_ROI = None   # 8x8 Grid alanı [x_start_ratio, y_start_ratio, width_ratio, height_ratio]
PIECES_ROI = None  # Alt kısımdaki 3 parça alanı [x_start_ratio, y_start_ratio, width_ratio, height_ratio]


def validate_roi(roi: object, name: str = "ROI") -> Tuple[bool, str]:
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return False, f"{name} must contain four normalized values"
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in roi):
        return False, f"{name} values must be numbers"

    x, y, width, height = (float(value) for value in roi)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return False, f"{name} values must be finite"
    if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
        return False, f"{name} origin must be inside the frame"
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return False, f"{name} dimensions must be positive"
    if x + width > 1.0 + 1e-9 or y + height > 1.0 + 1e-9:
        return False, f"{name} must fit inside the frame"
    return True, ""


def validate_calibration(
    board_roi: object,
    pieces_roi: object,
) -> Tuple[bool, str]:
    for roi, name in ((board_roi, "BOARD_ROI"), (pieces_roi, "PIECES_ROI")):
        valid, reason = validate_roi(roi, name)
        if not valid:
            return False, reason
    return True, ""


def _normalized_roi(roi: Sequence[float]) -> list[float]:
    return [float(value) for value in roi]


def save_calibration(board_roi: Sequence[float], pieces_roi: Sequence[float]) -> bool:
    """
    Kullanıcı tarafından seçilen ROI koordinat oranlarını JSON formatında diske kaydeder.
    Bu sayede uygulama her yeniden başlatıldığında kalibrasyon adımı atlanabilir.
    """
    valid, reason = validate_calibration(board_roi, pieces_roi)
    if not valid:
        logger.error("Calibration was not saved: %s", reason)
        return False

    data = {
        "BOARD_ROI": _normalized_roi(board_roi),
        "PIECES_ROI": _normalized_roi(pieces_roi),
    }
    calibration_path = Path(CALIBRATION_FILE)
    temporary_path = calibration_path.with_suffix(calibration_path.suffix + ".tmp")
    try:
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as calibration_handle:
            json.dump(data, calibration_handle, indent=4)
        temporary_path.replace(calibration_path)
    except (OSError, TypeError, ValueError) as error:
        temporary_path.unlink(missing_ok=True)
        logger.error("Calibration could not be saved: %s", error)
        return False

    global BOARD_ROI, PIECES_ROI
    BOARD_ROI = data["BOARD_ROI"]
    PIECES_ROI = data["PIECES_ROI"]
    logger.info("Calibration saved to %s", CALIBRATION_FILE)
    return True


def load_calibration():
    """
    Diskteki JSON kalibrasyon dosyasını okur ve ROI koordinat oranlarını yükler.
    Dosya bulunamazsa veya hata oluşursa None döndürür.
    """
    global BOARD_ROI, PIECES_ROI
    BOARD_ROI = None
    PIECES_ROI = None
    if not os.path.exists(CALIBRATION_FILE):
        logger.warning("Calibration file not found; calibration is required")
        return False

    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as calibration_handle:
            data = json.load(calibration_handle)
        if not isinstance(data, dict):
            raise ValueError("calibration root must be an object")

        board_roi = data.get("BOARD_ROI")
        pieces_roi = data.get("PIECES_ROI")
        valid, reason = validate_calibration(board_roi, pieces_roi)
        if not valid:
            raise ValueError(reason)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        logger.error("Calibration could not be loaded: %s", error)
        return False

    BOARD_ROI = _normalized_roi(board_roi)
    PIECES_ROI = _normalized_roi(pieces_roi)
    logger.info("Calibration loaded")
    return True
