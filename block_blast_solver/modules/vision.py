from typing import List, Optional, Tuple

import cv2
import numpy as np

from block_blast_solver import config
from block_blast_solver.modules import vision_models

UNIFORM_RANGE_THRESHOLD = 4.0
UNIFORM_STD_THRESHOLD = 1.5
UNIFORM_EMPTY_MAX_VALUE = 120.0
MIN_OCCUPIED_BRIGHTNESS_GAP = 18.0

# =====================================================================
# BLOCK BLAST SOLVER - GÖRÜNTÜ İŞLEME VE DİJİTAL HALE GETİRME (vision.py)
# =====================================================================

def _get_board_state_heuristic(frame: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    BOARD_ROI koordinatlarını kullanarak 8x8 tahta durumunu okur.
    Tema bağımsızlığı için HSV parlaklık kümelerini karşılaştırır.
    Döndürülen veri: (8x8 numpy dizisi (0=Boş, 1=Dolu), occlusion_flag)
    """
    if config.BOARD_ROI is None:
        return np.zeros((8, 8), dtype=np.uint8), False

    h, w, _ = frame.shape
    valid_roi, _ = config.validate_roi(config.BOARD_ROI, "BOARD_ROI")
    if not valid_roi:
        return np.zeros((8, 8), dtype=np.uint8), True
    
    # ROI koordinatlarını piksel cinsine çevir
    bx = max(0, int(config.BOARD_ROI[0] * w))
    by = max(0, int(config.BOARD_ROI[1] * h))
    bw = min(w - bx, int(config.BOARD_ROI[2] * w))
    bh = min(h - by, int(config.BOARD_ROI[3] * h))

    # Tahta görüntüsünü kes ve parlaklık kanalını çıkar
    board_crop = frame[by:by+bh, bx:bx+bw]
    if board_crop.size == 0:
        return np.zeros((8, 8), dtype=np.uint8), True

    hsv_board = cv2.cvtColor(board_crop, cv2.COLOR_BGR2HSV)
    value_board = hsv_board[:, :, 2]
    
    cell_w = bw / 8.0
    cell_h = bh / 8.0
    board_state = np.zeros((8, 8), dtype=np.uint8)

    cell_stats = []
    
    # 64 hücreyi tek tek incele
    for row in range(8):
        for col in range(8):
            # Hücrenin sol-üst ve sağ-alt sınırlarını hesapla
            c_x1 = int(col * cell_w)
            c_y1 = int(row * cell_h)
            c_w = int(cell_w)
            c_h = int(cell_h)

            # Kenar çizgilerini ve gölgeleri elemek için hücrenin yalnızca merkez %60'lık alanını incele
            offset_x = int(c_w * 0.20)
            offset_y = int(c_h * 0.20)
            inner_w = int(c_w * 0.60)
            inner_h = int(c_h * 0.60)

            # Hücre içi bölgeyi kes
            cell_inner = value_board[c_y1+offset_y:c_y1+offset_y+inner_h, c_x1+offset_x:c_x1+offset_x+inner_w]
            
            if cell_inner.size == 0:
                board_state[row, col] = 0
                continue

            mean_value = float(np.mean(cell_inner))
            cell_stats.append((row, col, mean_value))

    if not cell_stats:
        return board_state, True

    values = np.array([stat[2] for stat in cell_stats], dtype=np.float32)
    value_range = float(np.max(values) - np.min(values))
    sorted_values = np.sort(values)

    # A uniformly dark board is a valid empty-board state. A uniformly bright
    # crop is more likely to be a dialog, hand, or incorrect ROI.
    if value_range < UNIFORM_RANGE_THRESHOLD and float(np.std(values)) < UNIFORM_STD_THRESHOLD:
        if float(np.median(values)) <= UNIFORM_EMPTY_MAX_VALUE:
            return board_state, False
        return board_state, True

    # Empty board cells are the darkest stable cluster. Some occupied green tiles are
    # much dimmer than white tiles, so prefer the first meaningful gap rather than
    # the largest one. This also handles nearly full boards with very few empty cells.
    gaps = np.diff(sorted_values)
    meaningful_gaps = np.flatnonzero(gaps >= MIN_OCCUPIED_BRIGHTNESS_GAP)
    if meaningful_gaps.size:
        split_index = int(meaningful_gaps[0])
        threshold = float((sorted_values[split_index] + sorted_values[split_index + 1]) * 0.5)
    else:
        empty_sample_count = max(4, min(12, len(sorted_values) // 8))
        empty_values = sorted_values[:empty_sample_count]
        empty_baseline = float(np.median(empty_values))
        empty_noise = float(np.median(np.abs(empty_values - empty_baseline)))
        threshold = empty_baseline + max(24.0, empty_noise * 3.0 + 12.0)

    for row, col, mean_value in cell_stats:
        board_state[row, col] = 1 if mean_value >= threshold else 0

    return board_state, False


def get_board_state(frame: np.ndarray) -> Tuple[np.ndarray, bool]:
    if config.vision_force_heuristic():
        return _get_board_state_heuristic(frame)

    registry = vision_models.ModelRegistry.get()
    classifier = registry.board_classifier
    if classifier is None:
        return _get_board_state_heuristic(frame)

    if config.BOARD_ROI is None:
        return np.zeros((8, 8), dtype=np.uint8), False

    try:
        h, w, _ = frame.shape
        valid_roi, _ = config.validate_roi(config.BOARD_ROI, "BOARD_ROI")
        if not valid_roi:
            return np.zeros((8, 8), dtype=np.uint8), True

        bx = max(0, int(config.BOARD_ROI[0] * w))
        by = max(0, int(config.BOARD_ROI[1] * h))
        bw = min(w - bx, int(config.BOARD_ROI[2] * w))
        bh = min(h - by, int(config.BOARD_ROI[3] * h))

        board_crop = frame[by:by + bh, bx:bx + bw]
        if board_crop.size == 0:
            return np.zeros((8, 8), dtype=np.uint8), True

        cell_w = bw / 8.0
        cell_h = bh / 8.0
        probs = np.zeros((8, 8), dtype=np.float32)

        for row in range(8):
            for col in range(8):
                c_x1 = int(col * cell_w)
                c_y1 = int(row * cell_h)
                c_w = int(cell_w)
                c_h = int(cell_h)

                offset_x = int(c_w * 0.20)
                offset_y = int(c_h * 0.20)
                inner_w = int(c_w * 0.60)
                inner_h = int(c_h * 0.60)
                if inner_w <= 0 or inner_h <= 0:
                    raise ValueError("board cell inner crop is empty")

                cell_inner = board_crop[
                    c_y1 + offset_y:c_y1 + offset_y + inner_h,
                    c_x1 + offset_x:c_x1 + offset_x + inner_w,
                ]
                if cell_inner.size == 0:
                    raise ValueError("board cell crop is empty")

                probs[row, col] = classifier.predict_proba(cell_inner)
    except Exception:
        return np.zeros((8, 8), dtype=np.uint8), True

    if vision_models.board_probs_are_occluded(probs):
        return np.zeros((8, 8), dtype=np.uint8), True

    board_state = (probs >= config.T_BOARD).astype(np.uint8)
    return board_state, False


def _estimate_slot_background(slot_crop: np.ndarray) -> np.ndarray:
    h, w, _ = slot_crop.shape
    strip = max(2, min(h, w) // 12)
    samples = [
        slot_crop[:strip, :, :],
        slot_crop[h-strip:h, :, :],
        slot_crop[:, :strip, :],
        slot_crop[:, w-strip:w, :],
    ]
    border_pixels = np.concatenate([sample.reshape(-1, 3) for sample in samples], axis=0)
    return np.median(border_pixels, axis=0).astype(np.uint8)


def _build_piece_masks(slot_crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray_slot = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2GRAY)
    hsv_slot = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2HSV)
    lab_slot = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg_bgr = _estimate_slot_background(slot_crop)
    bg_pixel = np.uint8([[bg_bgr]])
    bg_hsv = cv2.cvtColor(bg_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    bg_lab = cv2.cvtColor(bg_pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    color_distance = np.sqrt(np.sum((lab_slot - bg_lab) ** 2, axis=2))
    saturation = hsv_slot[:, :, 1]
    value = hsv_slot[:, :, 2]
    bg_saturation = int(bg_hsv[1])
    bg_value = int(bg_hsv[2])
    bright_enough = value >= max(25, bg_value - 12)

    color_mask = np.zeros(gray_slot.shape, dtype=np.uint8)
    color_mask[((color_distance > 24.0) & bright_enough) | ((saturation > bg_saturation + 22) & (value >= max(35, bg_value - 24)))] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    edges = cv2.Canny(gray_slot, 50, 120)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)
    return color_mask, edge_mask


def _candidate_piece_bbox(color_mask: np.ndarray, edge_mask: np.ndarray, min_block_area: float) -> Optional[Tuple[int, int, int, int]]:
    def bbox_from_mask(mask: np.ndarray, min_area: float) -> Optional[Tuple[int, int, int, int]]:
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
        min_component_area = max(8, int(min_area * 0.08))

        for label in range(1, component_count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_component_area:
                candidate_mask[labels == label] = 255

        points = cv2.findNonZero(candidate_mask)
        if points is None:
            return None
        return cv2.boundingRect(points)

    bbox = bbox_from_mask(color_mask, min_block_area)
    if bbox is not None:
        return bbox
    return bbox_from_mask(edge_mask, min_block_area)


def _trim_empty_edges(piece_matrix: np.ndarray) -> np.ndarray:
    rows = np.where(np.any(piece_matrix == 1, axis=1))[0]
    cols = np.where(np.any(piece_matrix == 1, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return piece_matrix
    return piece_matrix[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _estimate_piece_grid_dims(bbox_w: int, bbox_h: int, inv_cell_w: float, inv_cell_h: float) -> Tuple[int, int]:
    target_cell = max(1.0, (inv_cell_w + inv_cell_h) * 0.5)
    best_rows = 1
    best_cols = 1
    best_score = float("inf")

    for rows in range(1, 6):
        cell_h_est = bbox_h / float(rows)
        if cell_h_est < 4.0:
            continue

        for cols in range(1, 6):
            cell_w_est = bbox_w / float(cols)
            if cell_w_est < 4.0:
                continue

            square_penalty = abs(cell_w_est - cell_h_est)
            size_penalty = abs(cell_w_est - target_cell) + abs(cell_h_est - target_cell)
            score = square_penalty * 8.0 + size_penalty

            if score < best_score:
                best_score = score
                best_rows = rows
                best_cols = cols

    return best_rows, best_cols


def _piece_from_mask(mask: np.ndarray, inv_cell_w: float, inv_cell_h: float) -> Optional[np.ndarray]:
    mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim != 2:
        raise ValueError("inventory mask must be a 2D array")

    mask_bin = ((mask_array >= config.T_MASK).astype(np.uint8)) * 255
    min_block_area = max(25.0, inv_cell_w * inv_cell_h * 0.15)
    if cv2.countNonZero(mask_bin) < min_block_area:
        return None

    points = cv2.findNonZero(mask_bin)
    if points is None:
        return None

    bx, by, bw, bh = cv2.boundingRect(points)
    num_rows, num_cols = _estimate_piece_grid_dims(bw, bh, inv_cell_w, inv_cell_h)
    piece_matrix = np.zeros((num_rows, num_cols), dtype=np.uint8)
    sub_w = bw / num_cols
    sub_h = bh / num_rows

    for r in range(num_rows):
        for c in range(num_cols):
            sc_x = int(bx + c * sub_w)
            sc_y = int(by + r * sub_h)
            sc_w = int(sub_w)
            sc_h = int(sub_h)

            if sc_w <= 0 or sc_h <= 0:
                continue

            inner_x = sc_x + int(sc_w * 0.22)
            inner_y = sc_y + int(sc_h * 0.22)
            inner_w = max(1, int(sc_w * 0.56))
            inner_h = max(1, int(sc_h * 0.56))

            mask_cell = mask_bin[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]
            mask_ratio = np.mean(mask_cell) / 255.0 if mask_cell.size > 0 else 0.0
            if mask_ratio > config.T_CELL:
                piece_matrix[r, c] = 1

    piece_matrix = _trim_empty_edges(piece_matrix)
    if not np.any(piece_matrix == 1):
        return None
    return piece_matrix


def _get_pieces_heuristic(frame: np.ndarray, cell_w: float, cell_h: float) -> List[Optional[np.ndarray]]:
    """
    PIECES_ROI koordinatlarını kullanarak inventory'deki 3 parçayı tespit eder.
    Her bir slotu binarize edip en büyük konturu bulur, ardından tahtanın hücre boyutuna
    göre oranlayarak parçanın binary matris şeklini çıkarır.
    """
    pieces = [None, None, None]
    if config.PIECES_ROI is None or cell_w <= 0 or cell_h <= 0:
        return pieces
    valid_roi, _ = config.validate_roi(config.PIECES_ROI, "PIECES_ROI")
    if not valid_roi:
        return pieces

    h, w, _ = frame.shape
    px = max(0, int(config.PIECES_ROI[0] * w))
    py = max(0, int(config.PIECES_ROI[1] * h))
    pw = min(w - px, int(config.PIECES_ROI[2] * w))
    ph = min(h - py, int(config.PIECES_ROI[3] * h))

    pieces_crop = frame[py:py+ph, px:px+pw]
    if pieces_crop.size == 0:
        return pieces

    # Envanter alanını yatayda 3 eşit bölgeye (Sol, Orta, Sağ) bölüyoruz
    slot_w = pw / 3.0
    
    for i in range(3):
        slot_x1 = int(i * slot_w)
        slot_x2 = int((i + 1) * slot_w)
        
        # Her bir slotu kes
        slot_crop = pieces_crop[0:ph, slot_x1:slot_x2]
        if slot_crop.size == 0:
            continue

        scale_factor = getattr(config, "PIECE_SCALE_FACTOR", 0.47)
        inv_cell_w = cell_w * scale_factor
        inv_cell_h = cell_h * scale_factor
        min_block_area = max(25.0, inv_cell_w * inv_cell_h * 0.15)

        color_mask, edge_mask = _build_piece_masks(slot_crop)
        piece_bbox = _candidate_piece_bbox(color_mask, edge_mask, min_block_area)

        # Eğer bulunan alan çok küçükse slot boştur
        if piece_bbox is None:
            continue

        bx, by, bw, bh = piece_bbox

        # Inventory blocks can render smaller than the board-derived scale, so infer
        # the logical grid by preferring square cells and only using board scale as a prior.
        num_rows, num_cols = _estimate_piece_grid_dims(bw, bh, inv_cell_w, inv_cell_h)

        # Parçanın matris formunu oluşturmak için bounding box'ı grid hücrelerine böl
        piece_matrix = np.zeros((num_rows, num_cols), dtype=np.uint8)
        sub_w = bw / num_cols
        sub_h = bh / num_rows

        for r in range(num_rows):
            for c in range(num_cols):
                # Her bir alt hücrenin koordinatı
                sc_x = int(bx + c * sub_w)
                sc_y = int(by + r * sub_h)
                sc_w = int(sub_w)
                sc_h = int(sub_h)

                if sc_w <= 0 or sc_h <= 0:
                    continue

                inner_x = sc_x + int(sc_w * 0.22)
                inner_y = sc_y + int(sc_h * 0.22)
                inner_w = max(1, int(sc_w * 0.56))
                inner_h = max(1, int(sc_h * 0.56))

                color_cell = color_mask[inner_y:inner_y+inner_h, inner_x:inner_x+inner_w]
                color_ratio = np.mean(color_cell) / 255.0 if color_cell.size > 0 else 0.0

                # Use color-only threshold. The old standalone edge_ratio condition
                # caused false-positives on wooden/textured backgrounds where Canny
                # edges bleed into empty corners of L/S/Z-shaped pieces.
                if color_ratio > 0.30:
                    piece_matrix[r, c] = 1

        piece_matrix = _trim_empty_edges(piece_matrix)
        if np.any(piece_matrix == 1):
            pieces[i] = piece_matrix

    return pieces


def get_pieces(frame: np.ndarray, cell_w: float, cell_h: float) -> List[Optional[np.ndarray]]:
    pieces = [None, None, None]
    if config.vision_force_heuristic():
        return _get_pieces_heuristic(frame, cell_w, cell_h)

    registry = vision_models.ModelRegistry.get()
    masker = registry.inventory_masker
    if masker is None:
        return _get_pieces_heuristic(frame, cell_w, cell_h)

    if config.PIECES_ROI is None or cell_w <= 0 or cell_h <= 0:
        return pieces
    valid_roi, _ = config.validate_roi(config.PIECES_ROI, "PIECES_ROI")
    if not valid_roi:
        return pieces

    h, w, _ = frame.shape
    px = max(0, int(config.PIECES_ROI[0] * w))
    py = max(0, int(config.PIECES_ROI[1] * h))
    pw = min(w - px, int(config.PIECES_ROI[2] * w))
    ph = min(h - py, int(config.PIECES_ROI[3] * h))

    pieces_crop = frame[py:py + ph, px:px + pw]
    if pieces_crop.size == 0:
        return pieces

    scale_factor = getattr(config, "PIECE_SCALE_FACTOR", 0.47)
    inv_cell_w = cell_w * scale_factor
    inv_cell_h = cell_h * scale_factor
    slot_w = pw / 3.0

    try:
        for i in range(3):
            slot_x1 = int(i * slot_w)
            slot_x2 = int((i + 1) * slot_w)
            slot_crop = pieces_crop[0:ph, slot_x1:slot_x2]
            if slot_crop.size == 0:
                continue

            slot_mask = masker.predict_mask(slot_crop)
            pieces[i] = _piece_from_mask(slot_mask, inv_cell_w, inv_cell_h)
    except Exception:
        return [None, None, None]

    return pieces
