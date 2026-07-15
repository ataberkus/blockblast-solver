import numpy as np
from numba import njit
from typing import List, Optional, Tuple, Dict, Any

from block_blast_solver import config

FUTURE_SET_PERMUTATIONS = np.array(
    [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ],
    dtype=np.int32,
)

# =====================================================================
# BLOCK BLAST SOLVER - NUMBA HIZLANDIRILMIŞ YAPAY ZEKA ÇÖZÜCÜ (solver.py)
# =====================================================================

@njit(cache=True)
def popcount(x: int) -> int:
    """
    64-bitlik tamsayıda (uint64) set edilmiş bit sayısını (Hamming ağırlığı) hesaplar.
    Hücre doluluk oranını bulmak için hızlı popcount algoritması kullanır.
    """
    u = np.uint64(x)
    u -= (u >> np.uint64(1)) & np.uint64(0x5555555555555555)
    u = (u & np.uint64(0x3333333333333333)) + ((u >> np.uint64(2)) & np.uint64(0x3333333333333333))
    u = (u + (u >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    u = (u * np.uint64(0x0101010101010101)) >> np.uint64(56)
    return int(u)


@njit(cache=True)
def calculate_holes_jit(board_mask: int) -> int:
    """
    Üzerinde en az bir dolu hücre bulunan sıkışmış boş hücre sayısını hesaplar.
    Doldurulması en zor olan hücreleri tespit eder. Bit düzeyinde optimize edilmiştir.
    """
    holes = 0
    u_board = np.uint64(board_mask)
    for c in range(8):
        block_found = False
        for r in range(8):
            bit_idx = np.uint64(r * 8 + c)
            if (u_board & (np.uint64(1) << bit_idx)) != np.uint64(0):
                block_found = True
            elif block_found:
                holes += 1
    return holes


@njit(cache=True)
def calculate_bumpiness_jit(board_mask: int) -> int:
    """
    Yan yana olan sütunların yükseklik farklarının mutlak değerlerinin toplamını bulur.
    Düz bir yüzey tutmak oyunun devamlılığı için kritiktir.
    Sıfır bellek tahsisiyle bit düzeyinde çalışır.
    """
    bumpiness = 0
    prev_h = -1
    u_board = np.uint64(board_mask)
    for c in range(8):
        h = 0
        for r in range(8):
            bit_idx = np.uint64(r * 8 + c)
            if (u_board & (np.uint64(1) << bit_idx)) != np.uint64(0):
                h = 8 - r
                break
        if prev_h != -1:
            bumpiness += abs(prev_h - h)
        prev_h = h
    return bumpiness


@njit(cache=True)
def calculate_empty_regions_jit(board_mask: int) -> Tuple[int, int, int]:
    """
    Boş hücrelerin bağlantılı bölgelerini hesaplar.
    Dönen değerler: (en büyük boş bölge, küçük bölge sayısı, toplam bölge sayısı)
    """
    u_board = np.uint64(board_mask)
    visited = np.uint64(0)
    largest_region = 0
    small_regions = 0
    region_count = 0

    for start in range(64):
        start_bit = np.uint64(1) << np.uint64(start)
        if (u_board & start_bit) != np.uint64(0) or (visited & start_bit) != np.uint64(0):
            continue

        region_count += 1
        stack = np.zeros(64, dtype=np.int32)
        stack_size = 1
        stack[0] = start
        visited |= start_bit
        region_size = 0

        while stack_size > 0:
            stack_size -= 1
            idx = stack[stack_size]
            region_size += 1
            row = idx // 8
            col = idx % 8

            if row > 0:
                next_idx = idx - 8
                next_bit = np.uint64(1) << np.uint64(next_idx)
                if (u_board & next_bit) == np.uint64(0) and (visited & next_bit) == np.uint64(0):
                    visited |= next_bit
                    stack[stack_size] = next_idx
                    stack_size += 1

            if row < 7:
                next_idx = idx + 8
                next_bit = np.uint64(1) << np.uint64(next_idx)
                if (u_board & next_bit) == np.uint64(0) and (visited & next_bit) == np.uint64(0):
                    visited |= next_bit
                    stack[stack_size] = next_idx
                    stack_size += 1

            if col > 0:
                next_idx = idx - 1
                next_bit = np.uint64(1) << np.uint64(next_idx)
                if (u_board & next_bit) == np.uint64(0) and (visited & next_bit) == np.uint64(0):
                    visited |= next_bit
                    stack[stack_size] = next_idx
                    stack_size += 1

            if col < 7:
                next_idx = idx + 1
                next_bit = np.uint64(1) << np.uint64(next_idx)
                if (u_board & next_bit) == np.uint64(0) and (visited & next_bit) == np.uint64(0):
                    visited |= next_bit
                    stack[stack_size] = next_idx
                    stack_size += 1

        if region_size > largest_region:
            largest_region = region_size
        if region_size <= 3:
            small_regions += 1

    return largest_region, small_regions, region_count


@njit(cache=True)
def calculate_trap_penalty_jit(board_mask: int) -> int:
    """
    Üç yanı kapalı veya tek çıkışlı boş hücreleri cezalandırmak için sayar.
    """
    u_board = np.uint64(board_mask)
    traps = 0

    for r in range(8):
        for c in range(8):
            idx = r * 8 + c
            bit = np.uint64(1) << np.uint64(idx)
            if (u_board & bit) != np.uint64(0):
                continue

            blocked_neighbors = 0
            open_neighbors = 0

            if r == 0 or (u_board & (np.uint64(1) << np.uint64(idx - 8))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1

            if r == 7 or (u_board & (np.uint64(1) << np.uint64(idx + 8))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1

            if c == 0 or (u_board & (np.uint64(1) << np.uint64(idx - 1))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1

            if c == 7 or (u_board & (np.uint64(1) << np.uint64(idx + 1))) != np.uint64(0):
                blocked_neighbors += 1
            else:
                open_neighbors += 1

            if blocked_neighbors >= 3:
                traps += 2
            elif open_neighbors == 1:
                traps += 1

    return traps


@njit(cache=True)
def calculate_line_readiness_survival_jit(board_mask: int) -> int:
    """
    Temizlenmeye yakın satır ve sütunları geleceğe dönük fırsat olarak puanlar.
    """
    u_board = np.uint64(board_mask)
    readiness = 0

    for r in range(8):
        row_mask = np.uint64(0xFF) << np.uint64(r * 8)
        filled = popcount(u_board & row_mask)
        missing = 8 - filled
        if missing == 1:
            readiness += 6
        elif missing == 2:
            readiness += 3
        elif missing == 3:
            readiness += 1

    for c in range(8):
        col_mask = np.uint64(0x0101010101010101) << np.uint64(c)
        filled = popcount(u_board & col_mask)
        missing = 8 - filled
        if missing == 1:
            readiness += 6
        elif missing == 2:
            readiness += 3
        elif missing == 3:
            readiness += 1

    return readiness


@njit(cache=True)
def get_monte_carlo_piece_catalog() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique piece orientations used by future-set sampling."""
    masks = np.zeros(25, dtype=np.uint64)
    shapes = np.zeros((25, 2), dtype=np.int32)
    families = np.zeros(25, dtype=np.int32)

    masks[0] = np.uint64(0x1); shapes[0, 0] = 1; shapes[0, 1] = 1; families[0] = 1
    masks[1] = np.uint64(0x3); shapes[1, 0] = 1; shapes[1, 1] = 2; families[1] = 2
    masks[2] = np.uint64(0x101); shapes[2, 0] = 2; shapes[2, 1] = 1; families[2] = 2
    masks[3] = np.uint64(0x7); shapes[3, 0] = 1; shapes[3, 1] = 3; families[3] = 3
    masks[4] = np.uint64(0x10101); shapes[4, 0] = 3; shapes[4, 1] = 1; families[4] = 3
    masks[5] = np.uint64(0xF); shapes[5, 0] = 1; shapes[5, 1] = 4; families[5] = 4
    masks[6] = np.uint64(0x1010101); shapes[6, 0] = 4; shapes[6, 1] = 1; families[6] = 4
    masks[7] = np.uint64(0x1F); shapes[7, 0] = 1; shapes[7, 1] = 5; families[7] = 5
    masks[8] = np.uint64(0x101010101); shapes[8, 0] = 5; shapes[8, 1] = 1; families[8] = 5
    masks[9] = np.uint64(0x303); shapes[9, 0] = 2; shapes[9, 1] = 2; families[9] = 9
    masks[10] = np.uint64(0x70707); shapes[10, 0] = 3; shapes[10, 1] = 3; families[10] = 25

    masks[11] = np.uint64(0x107); shapes[11, 0] = 2; shapes[11, 1] = 3; families[11] = 11
    masks[12] = np.uint64(0x701); shapes[12, 0] = 2; shapes[12, 1] = 3; families[12] = 11
    masks[13] = np.uint64(0x30102); shapes[13, 0] = 3; shapes[13, 1] = 2; families[13] = 11
    masks[14] = np.uint64(0x20103); shapes[14, 0] = 3; shapes[14, 1] = 2; families[14] = 11
    masks[15] = np.uint64(0x407); shapes[15, 0] = 2; shapes[15, 1] = 3; families[15] = 11
    masks[16] = np.uint64(0x704); shapes[16, 0] = 2; shapes[16, 1] = 3; families[16] = 11
    masks[17] = np.uint64(0x10203); shapes[17, 0] = 3; shapes[17, 1] = 2; families[17] = 11
    masks[18] = np.uint64(0x30201); shapes[18, 0] = 3; shapes[18, 1] = 2; families[18] = 11

    masks[19] = np.uint64(0x207); shapes[19, 0] = 2; shapes[19, 1] = 3; families[19] = 19
    masks[20] = np.uint64(0x702); shapes[20, 0] = 2; shapes[20, 1] = 3; families[20] = 19

    masks[21] = np.uint64(0x306); shapes[21, 0] = 2; shapes[21, 1] = 3; families[21] = 23
    masks[22] = np.uint64(0x603); shapes[22, 0] = 2; shapes[22, 1] = 3; families[22] = 23
    masks[23] = np.uint64(0x10206); shapes[23, 0] = 3; shapes[23, 1] = 2; families[23] = 23

    masks[24] = np.uint64(0x707); shapes[24, 0] = 2; shapes[24, 1] = 3; families[24] = 27
    return masks, shapes, families


@njit(cache=True)
def get_catalog_cumulative_weights(families: np.ndarray) -> Tuple[np.ndarray, int]:
    """Weight families equally while distributing weight across orientations."""
    family_counts = np.zeros(32, dtype=np.int32)
    for family in families:
        family_counts[family] += 1

    cumulative_weights = np.zeros(families.shape[0], dtype=np.int32)
    total_weight = 0
    for index in range(families.shape[0]):
        family_count = family_counts[families[index]]
        weight = 2520 // max(1, family_count)
        total_weight += weight
        cumulative_weights[index] = total_weight
    return cumulative_weights, total_weight


@njit(cache=True)
def _next_random_state(state: np.uint64) -> np.uint64:
    return state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)


@njit(cache=True)
def _weighted_catalog_index(ticket: int, cumulative_weights: np.ndarray) -> int:
    for index in range(cumulative_weights.shape[0]):
        if ticket < cumulative_weights[index]:
            return index
    return cumulative_weights.shape[0] - 1


@njit(cache=True)
def get_next_piece_samples(sample_count: int, board_mask: int = 0, seed: int = 0xC0FFEE) -> np.ndarray:
    masks, _, families = get_monte_carlo_piece_catalog()
    catalog_size = masks.shape[0]
    safe_sample_count = max(0, sample_count)
    samples = np.zeros((safe_sample_count, 3), dtype=np.int32)
    if catalog_size == 0 or safe_sample_count == 0:
        return samples

    cumulative_weights, total_weight = get_catalog_cumulative_weights(families)
    state = np.uint64(board_mask) ^ np.uint64(seed) ^ np.uint64(safe_sample_count * 0x9E3779B9)
    for sample_idx in range(safe_sample_count):
        for slot in range(3):
            state = _next_random_state(state)
            ticket = int((state >> np.uint64(32)) % np.uint64(total_weight))
            samples[sample_idx, slot] = _weighted_catalog_index(ticket, cumulative_weights)

    return samples


@njit(cache=True)
def get_future_piece_masks() -> Tuple[np.ndarray, np.ndarray]:
    masks = np.zeros(16, dtype=np.uint64)
    shapes = np.zeros((16, 2), dtype=np.int32)

    masks[0] = np.uint64(0x1); shapes[0, 0] = 1; shapes[0, 1] = 1
    masks[1] = np.uint64(0x3); shapes[1, 0] = 1; shapes[1, 1] = 2
    masks[2] = np.uint64(0x101); shapes[2, 0] = 2; shapes[2, 1] = 1
    masks[3] = np.uint64(0x7); shapes[3, 0] = 1; shapes[3, 1] = 3
    masks[4] = np.uint64(0x10101); shapes[4, 0] = 3; shapes[4, 1] = 1
    masks[5] = np.uint64(0xF); shapes[5, 0] = 1; shapes[5, 1] = 4
    masks[6] = np.uint64(0x1010101); shapes[6, 0] = 4; shapes[6, 1] = 1
    masks[7] = np.uint64(0x1F); shapes[7, 0] = 1; shapes[7, 1] = 5
    masks[8] = np.uint64(0x101010101); shapes[8, 0] = 5; shapes[8, 1] = 1
    masks[9] = np.uint64(0x303); shapes[9, 0] = 2; shapes[9, 1] = 2
    masks[10] = np.uint64(0x70707); shapes[10, 0] = 3; shapes[10, 1] = 3
    masks[11] = np.uint64(0x107); shapes[11, 0] = 2; shapes[11, 1] = 3
    masks[12] = np.uint64(0x701); shapes[12, 0] = 2; shapes[12, 1] = 3
    masks[13] = np.uint64(0x30102); shapes[13, 0] = 3; shapes[13, 1] = 2
    masks[14] = np.uint64(0x20103); shapes[14, 0] = 3; shapes[14, 1] = 2
    masks[15] = np.uint64(0x702); shapes[15, 0] = 2; shapes[15, 1] = 3

    return masks, shapes


@njit(cache=True)
def count_future_piece_fits_jit(board_mask: int) -> int:
    """
    Yaygın gelecek parça ailelerinin kaç farklı esneklikle sığabildiğini ölçer.
    """
    future_masks, future_shapes = get_future_piece_masks()
    u_board = np.uint64(board_mask)
    fit_score = 0

    for i in range(future_masks.shape[0]):
        piece_mask = future_masks[i]
        ph = future_shapes[i, 0]
        pw = future_shapes[i, 1]
        placements = 0

        for r in range(9 - ph):
            for c in range(9 - pw):
                shifted_piece = piece_mask << np.uint64(r * 8 + c)
                if (u_board & shifted_piece) == np.uint64(0):
                    placements += 1

        if placements > 0:
            fit_score += 8
            if placements > 2:
                fit_score += 4
            if placements > 6:
                fit_score += 3

    return fit_score


@njit(cache=True)
def check_readiness_jit(board_mask: int, empty_cells: int) -> float:
    """
    Kritik büyük parçaların (3x3 kare, 5x1 yatay, 1x5 dikey) tahtaya
    sığıp sığamayacağını kontrol eder. Sığan parça başına puan verir.
    Bit düzeyinde maske kaydırma ile son derece hızlı çalışır.
    """
    u_board = np.uint64(board_mask)
    
    # 3x3 sığıyor mu kontrolü
    fit_3x3 = 0.0
    if empty_cells >= 9:
        mask_3x3_base = np.uint64(0x0000000000070707)
        for r in range(6):
            for c in range(6):
                shift = np.uint64(r * 8 + c)
                if (u_board & (mask_3x3_base << shift)) == np.uint64(0):
                    fit_3x3 = 1.0
                    break
            if fit_3x3 == 1.0:
                break

    # 5x1 sığıyor mu kontrolü (yatay çizgi)
    fit_5x1 = 0.0
    if empty_cells >= 5:
        mask_5x1_base = np.uint64(0x1F)
        for r in range(8):
            for c in range(4):
                shift = np.uint64(r * 8 + c)
                if (u_board & (mask_5x1_base << shift)) == np.uint64(0):
                    fit_5x1 = 1.0
                    break
            if fit_5x1 == 1.0:
                break

    # 1x5 sığıyor mu kontrolü (dikey çizgi)
    fit_1x5 = 0.0
    if empty_cells >= 5:
        mask_1x5_base = np.uint64(0x0000000101010101)
        for r in range(4):
            for c in range(8):
                shift = np.uint64(r * 8 + c)
                if (u_board & (mask_1x5_base << shift)) == np.uint64(0):
                    fit_1x5 = 1.0
                    break
            if fit_1x5 == 1.0:
                break

    return fit_3x3 + fit_5x1 + fit_1x5


@njit(cache=True)
def choose_monte_carlo_sample_count_jit(board_mask: int,
                                        normal_samples: int,
                                        danger_samples: int,
                                        danger_filled_cells: int,
                                        min_future_fits: int) -> int:
    filled_cells = popcount(board_mask)
    future_fits = count_future_piece_fits_jit(board_mask)
    readiness = check_readiness_jit(board_mask, 64 - filled_cells)
    _, _, region_count = calculate_empty_regions_jit(board_mask)

    if filled_cells >= danger_filled_cells:
        return danger_samples
    if future_fits <= min_future_fits:
        return danger_samples
    if readiness < 1.0:
        return danger_samples
    if region_count >= 4:
        return danger_samples
    return normal_samples


@njit(cache=True)
def evaluate_survival_score_jit(board_mask: int,
                                accumulated_score: float,
                                w_empty: float,
                                w_holes: float,
                                w_bumpiness: float,
                                w_readiness: float,
                                w_future_fits: float,
                                w_largest_region: float,
                                w_small_region_penalty: float,
                                w_line_readiness_survival: float,
                                w_trap_penalty: float,
                                normal_samples: int,
                                danger_samples: int,
                                danger_filled_cells: int,
                                min_future_fits: int,
                                monte_carlo_seed: int,
                                w_monte_carlo_survival: float,
                                w_monte_carlo_clear_routes: float,
                                w_monte_carlo_future_fits: float,
                                w_monte_carlo_streak: float) -> Tuple[float, float, int, int, int, float]:
    empty_cells = 64 - popcount(board_mask)
    holes = calculate_holes_jit(board_mask)
    bumpiness = calculate_bumpiness_jit(board_mask)
    readiness = check_readiness_jit(board_mask, empty_cells)
    future_fits = count_future_piece_fits_jit(board_mask)
    largest_region, small_regions, region_count = calculate_empty_regions_jit(board_mask)
    line_readiness = calculate_line_readiness_survival_jit(board_mask)
    traps = calculate_trap_penalty_jit(board_mask)
    fragmentation_penalty = max(0, region_count - 1) * 20.0

    sample_count = choose_monte_carlo_sample_count_jit(
        board_mask,
        normal_samples,
        danger_samples,
        danger_filled_cells,
        min_future_fits,
    )
    _, next_survival_pct, clear_routes, mc_future_fits, streak_continuity = evaluate_monte_carlo_survival_jit(
        board_mask,
        sample_count,
        monte_carlo_seed,
    )

    base_score = (
        (accumulated_score * 0.75)
        + (w_empty * empty_cells)
        + (w_holes * holes)
        + (w_bumpiness * bumpiness)
        + (w_readiness * readiness)
        + (w_future_fits * future_fits)
        + (w_largest_region * largest_region)
        + (w_small_region_penalty * small_regions)
        + (w_line_readiness_survival * line_readiness)
        + (w_trap_penalty * traps)
        - fragmentation_penalty
    )
    monte_carlo_score = (
        (next_survival_pct * w_monte_carlo_survival)
        + (clear_routes * w_monte_carlo_clear_routes)
        + (mc_future_fits * w_monte_carlo_future_fits)
        + (streak_continuity * w_monte_carlo_streak)
    )

    return base_score + monte_carlo_score, next_survival_pct, clear_routes, mc_future_fits, sample_count, streak_continuity


@njit(cache=True)
def clear_completed_lines_jit(board_mask: int, row_start: int, row_count: int, col_start: int, col_count: int) -> Tuple[int, int]:
    u_board = np.uint64(board_mask)
    row_clear_mask = np.uint64(0)
    col_clear_mask = np.uint64(0)
    clear_count = 0

    for row_offset in range(row_count):
        row_idx = row_start + row_offset
        if row_idx < 0 or row_idx >= 8:
            continue
        mask = np.uint64(0xFF) << np.uint64(row_idx * 8)
        if (u_board & mask) == mask:
            row_clear_mask |= mask
            clear_count += 1

    for col_offset in range(col_count):
        col_idx = col_start + col_offset
        if col_idx < 0 or col_idx >= 8:
            continue
        mask = np.uint64(0x0101010101010101) << np.uint64(col_idx)
        if (u_board & mask) == mask:
            col_clear_mask |= mask
            clear_count += 1

    if row_clear_mask != 0 or col_clear_mask != 0:
        u_board = u_board & ~(row_clear_mask | col_clear_mask)

    return int(np.int64(u_board)), clear_count


@njit(cache=True)
def simulate_best_single_future_piece_jit(board_mask: int, piece_mask: int, ph: int, pw: int) -> Tuple[bool, int, int]:
    u_board = np.uint64(board_mask)
    best_board = np.int64(0)
    best_score = -1000000000
    best_clears = 0
    found = False

    for row in range(9 - ph):
        for col in range(9 - pw):
            shifted_piece = np.uint64(piece_mask) << np.uint64(row * 8 + col)
            if (u_board & shifted_piece) != np.uint64(0):
                continue

            placed_board = np.int64(board_mask) | np.int64(shifted_piece)
            cleared_board, clear_count = clear_completed_lines_jit(placed_board, row, ph, col, pw)
            empty_cells = 64 - popcount(cleared_board)
            future_fits = count_future_piece_fits_jit(cleared_board)
            largest_region, small_regions, region_count = calculate_empty_regions_jit(cleared_board)
            score = (
                (clear_count * 1000)
                + (future_fits * 18)
                + (largest_region * 8)
                + (empty_cells * 5)
                - (small_regions * 80)
                - (max(0, region_count - 1) * 25)
            )

            if score > best_score:
                best_score = score
                best_board = cleared_board
                best_clears = clear_count
                found = True

    return found, int(best_board), best_clears


@njit(cache=True)
def simulate_next_set_one_order_jit(board_mask: int,
                                    sample: np.ndarray,
                                    catalog_masks: np.ndarray,
                                    catalog_shapes: np.ndarray) -> Tuple[bool, int, int, int, int]:
    current_board = board_mask
    total_clears = 0
    clearing_placements = 0

    for sample_pos in range(3):
        piece_idx = sample[sample_pos]
        piece_mask = int(catalog_masks[piece_idx])
        ph = int(catalog_shapes[piece_idx, 0])
        pw = int(catalog_shapes[piece_idx, 1])
        placed, next_board, clears = simulate_best_single_future_piece_jit(current_board, piece_mask, ph, pw)
        if not placed:
            return False, current_board, total_clears, sample_pos, clearing_placements
        current_board = next_board
        total_clears += clears
        if clears > 0:
            clearing_placements += 1

    return True, current_board, total_clears, 3, clearing_placements


@njit(cache=True)
def simulate_next_set_survival_jit(board_mask: int,
                                   sample: np.ndarray,
                                   catalog_masks: np.ndarray,
                                   catalog_shapes: np.ndarray) -> Tuple[bool, int, int, int, int]:
    """Try every piece order and retain the most playable greedy outcome."""
    best_survived = False
    best_board = board_mask
    best_clears = 0
    best_placed = 0
    best_clearing_placements = 0
    best_quality = -1
    ordered_sample = np.empty(3, dtype=np.int32)

    for permutation_index in range(FUTURE_SET_PERMUTATIONS.shape[0]):
        for sample_position in range(3):
            ordered_sample[sample_position] = sample[FUTURE_SET_PERMUTATIONS[permutation_index, sample_position]]

        survived, resulting_board, clears, placed, clearing_placements = simulate_next_set_one_order_jit(
            board_mask,
            ordered_sample,
            catalog_masks,
            catalog_shapes,
        )
        future_fits = count_future_piece_fits_jit(resulting_board)
        quality = (
            (1000000000 if survived else 0)
            + (placed * 1000000)
            + (clears * 10000)
            + (clearing_placements * 1000)
            + future_fits
        )
        if quality > best_quality:
            best_quality = quality
            best_survived = survived
            best_board = resulting_board
            best_clears = clears
            best_placed = placed
            best_clearing_placements = clearing_placements

    return best_survived, best_board, best_clears, best_placed, best_clearing_placements


@njit(cache=True)
def evaluate_monte_carlo_survival_jit(
    board_mask: int,
    sample_count: int,
    seed: int = 0xC0FFEE,
) -> Tuple[float, float, int, int, float]:
    if sample_count <= 0:
        return 0.0, 0.0, 0, 0, 0.0

    catalog_masks, catalog_shapes, _ = get_monte_carlo_piece_catalog()
    samples = get_next_piece_samples(sample_count, board_mask, seed)
    survived = 0
    total_clear_routes = 0
    total_future_fits = 0
    partial_progress = 0
    streak_ratio_sum = 0.0
    streak_ratio_count = 0

    for sample_idx in range(sample_count):
        sample = samples[sample_idx]
        did_survive, resulting_board, clears, placed_count, clearing_placements = simulate_next_set_survival_jit(
            board_mask,
            sample,
            catalog_masks,
            catalog_shapes,
        )
        if did_survive:
            survived += 1
        total_clear_routes += clears
        partial_progress += placed_count
        total_future_fits += count_future_piece_fits_jit(resulting_board)
        if placed_count > 0:
            streak_ratio_sum += clearing_placements / placed_count
            streak_ratio_count += 1

    survival_pct = (survived * 100.0) / sample_count
    average_clear_routes = total_clear_routes / sample_count
    average_future_fits = total_future_fits / sample_count
    average_progress = partial_progress / sample_count
    score = survival_pct + (average_clear_routes * 3.0) + (average_future_fits * 2.0) + (average_progress * 10.0)

    if streak_ratio_count > 0:
        avg_streak_continuity = streak_ratio_sum / streak_ratio_count
    else:
        avg_streak_continuity = 0.0

    return score, survival_pct, int(average_clear_routes + 0.5), int(average_future_fits + 0.5), avg_streak_continuity


@njit(cache=True)
def solve_recursive(board_mask: int,
                    depth: int,
                    max_depth: int,
                    pieces_masks: np.ndarray,
                    pieces_shapes: np.ndarray,
                    pieces_active: np.ndarray,
                    current_moves: np.ndarray,
                    best_moves_global: np.ndarray,
                    best_score_holder: np.ndarray,
                    best_diagnostics_holder: np.ndarray,
                    accumulated_score: float,
                    w_clear: float,
                    w_empty: float,
                    w_holes: float,
                    w_bumpiness: float,
                    w_readiness: float,
                    w_future_fits: float,
                    w_largest_region: float,
                    w_small_region_penalty: float,
                    w_line_readiness_survival: float,
                    w_trap_penalty: float,
                    normal_samples: int,
                    danger_samples: int,
                    danger_filled_cells: int,
                    min_future_fits: int,
                    monte_carlo_seed: int,
                    w_monte_carlo_survival: float,
                    w_monte_carlo_clear_routes: float,
                    w_monte_carlo_future_fits: float,
                    w_monte_carlo_streak: float,
                    w_streak_continue: float,
                    w_streak_break_penalty: float,
                    w_streak_perfect_bonus: float,
                    streak_in_plan: int,
                    node_budget: int,
                    nodes_visited_holder: np.ndarray,
                    budget_exhausted_holder: np.ndarray) -> None:
    """
    Derinlemesine arama (backtracking) yapan recursive JIT fonksiyonu.
    Maksimum hız için sıfır dinamik bellek tahsisi ve bit düzeyinde durum takibi kullanır.
    """
    if node_budget > 0 and nodes_visited_holder[0] >= node_budget:
        budget_exhausted_holder[0] = True
        return
    nodes_visited_holder[0] += 1

    if depth == max_depth:
        score, next_survival_pct, clear_routes, mc_future_fits, sample_count, streak_continuity = evaluate_survival_score_jit(
            board_mask,
            accumulated_score,
            w_empty,
            w_holes,
            w_bumpiness,
            w_readiness,
            w_future_fits,
            w_largest_region,
            w_small_region_penalty,
            w_line_readiness_survival,
            w_trap_penalty,
            normal_samples,
            danger_samples,
            danger_filled_cells,
            min_future_fits,
            monte_carlo_seed,
            w_monte_carlo_survival,
            w_monte_carlo_clear_routes,
            w_monte_carlo_future_fits,
            w_monte_carlo_streak,
        )
        
        if score > best_score_holder[0]:
            best_score_holder[0] = score
            best_diagnostics_holder[0] = next_survival_pct
            best_diagnostics_holder[1] = clear_routes
            best_diagnostics_holder[2] = mc_future_fits
            best_diagnostics_holder[3] = sample_count
            best_diagnostics_holder[4] = streak_continuity
            for d in range(max_depth):
                best_moves_global[d, 0] = current_moves[d, 0]
                best_moves_global[d, 1] = current_moves[d, 1]
                best_moves_global[d, 2] = current_moves[d, 2]
        return

    u_board = np.uint64(board_mask)
    # 3 parçayı sırayla dene
    for i in range(3):
        if pieces_active[i]:
            ph = pieces_shapes[i, 0]
            pw = pieces_shapes[i, 1]
            piece_mask = np.uint64(pieces_masks[i])

            # Tahtada sığabilecek tüm koordinatları dene
            for r in range(9 - ph):
                for c in range(9 - pw):
                    shift = np.uint64(r * 8 + c)
                    shifted_piece = piece_mask << shift
                    if (u_board & shifted_piece) == np.uint64(0):
                        # Yerleştir (OR işlemi ile)
                        new_board_mask = np.int64(board_mask) | np.int64(shifted_piece)
                        
                        # Dolan satır ve sütunları temizle
                        cleared_rows = 0
                        row_clear_mask = np.uint64(0)
                        for row_offset in range(ph):
                            row_idx = r + row_offset
                            mask = np.uint64(0xFF) << np.uint64(row_idx * 8)
                            if (np.uint64(new_board_mask) & mask) == mask:
                                row_clear_mask |= mask
                                cleared_rows += 1

                        cleared_cols = 0
                        col_clear_mask = np.uint64(0)
                        for col_offset in range(pw):
                            col_idx = c + col_offset
                            mask = np.uint64(0x0101010101010101) << np.uint64(col_idx)
                            if (np.uint64(new_board_mask) & mask) == mask:
                                col_clear_mask |= mask
                                cleared_cols += 1

                        if row_clear_mask != 0 or col_clear_mask != 0:
                            new_board_mask = np.int64(np.uint64(new_board_mask) & ~(row_clear_mask | col_clear_mask))

                        is_clearing = (cleared_rows + cleared_cols) > 0
                        if is_clearing:
                            new_streak = streak_in_plan + 1
                            streak_delta = w_streak_continue
                        else:
                            new_streak = 0
                            streak_delta = w_streak_break_penalty
                        if depth + 1 == max_depth and new_streak == max_depth:
                            streak_delta += w_streak_perfect_bonus

                        # Mevcut hamleyi kaydet
                        current_moves[depth, 0] = i
                        current_moves[depth, 1] = r
                        current_moves[depth, 2] = c

                        # Parçayı inaktif yap
                        pieces_active[i] = False
                        
                        # Bir sonraki derinliğe geç (tamsayı kopyalandığı için backtrack zahmetsizdir)
                        solve_recursive(
                            new_board_mask, depth + 1, max_depth,
                            pieces_masks, pieces_shapes, pieces_active,
                            current_moves, best_moves_global, best_score_holder, best_diagnostics_holder,
                            accumulated_score + ((cleared_rows + cleared_cols) * w_clear) + streak_delta,
                            w_clear, w_empty, w_holes, w_bumpiness, w_readiness,
                            w_future_fits, w_largest_region, w_small_region_penalty,
                            w_line_readiness_survival, w_trap_penalty,
                            normal_samples, danger_samples, danger_filled_cells, min_future_fits,
                            monte_carlo_seed,
                            w_monte_carlo_survival, w_monte_carlo_clear_routes, w_monte_carlo_future_fits,
                            w_monte_carlo_streak,
                            w_streak_continue, w_streak_break_penalty, w_streak_perfect_bonus,
                            new_streak,
                            node_budget,
                            nodes_visited_holder,
                            budget_exhausted_holder,
                        )

                        # Parçayı aktif yap (backtrack)
                        pieces_active[i] = True


def _diagnostics_from_holder(best_score: float, holder: np.ndarray) -> Dict[str, Any]:
    if best_score < -1e8:
        return {
            "risk_level": "dead",
            "next_survival_pct": 0.0,
            "clear_routes": 0,
            "future_fits": 0,
            "sample_count": 0,
            "next_streak_pct": 0.0,
        }

    next_survival_pct = float(holder[0])
    if next_survival_pct >= 70.0:
        risk_level = "low"
    elif next_survival_pct >= 35.0:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "risk_level": risk_level,
        "next_survival_pct": next_survival_pct,
        "clear_routes": int(holder[1]),
        "future_fits": int(holder[2]),
        "sample_count": int(holder[3]),
        "next_streak_pct": float(holder[4]),
    }


def _solve_core(
    board: np.ndarray,
    active_pieces: List[Optional[np.ndarray]],
    node_budget: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], float, Dict[str, Any]]:
    """
    Python tarafındaki sarmalayıcı (wrapper) arayüz.
    Numpy dizisi halindeki parçaları ve tahtayı bit maskelerine dönüştürür
    ve JIT çözücüyü başlatır.
    """
    if board.shape != (8, 8):
        raise ValueError("board must have shape (8, 8)")
    if len(active_pieces) > 3:
        raise ValueError("at most three active pieces are supported")
    if not np.all((board == 0) | (board == 1)):
        raise ValueError("board values must be binary")

    pieces_masks = np.zeros(3, dtype=np.uint64)
    pieces_shapes = np.zeros((3, 2), dtype=np.int32)
    pieces_active = np.zeros(3, dtype=bool)

    active_count = 0
    original_indices = [-1, -1, -1]

    # Parçaları bit maskelerine dönüştürme
    for idx, matrix in enumerate(active_pieces):
        if matrix is not None:
            if matrix.ndim != 2 or matrix.size == 0:
                raise ValueError(f"piece {idx} must be a non-empty 2D matrix")
            ph, pw = matrix.shape
            if ph > 8 or pw > 8:
                raise ValueError(f"piece {idx} must fit within an 8x8 board")
            if not np.all((matrix == 0) | (matrix == 1)) or not np.any(matrix == 1):
                raise ValueError(f"piece {idx} must contain a non-empty binary shape")
            pieces_shapes[active_count, 0] = ph
            pieces_shapes[active_count, 1] = pw
            
            piece_mask = np.uint64(0)
            for r in range(ph):
                for c in range(pw):
                    if matrix[r, c] == 1:
                        piece_mask |= np.uint64(1) << np.uint64(r * 8 + c)
            pieces_masks[active_count] = piece_mask
            pieces_active[active_count] = True
            original_indices[active_count] = idx
            active_count += 1

    if active_count == 0:
        return None, 0.0, {
            "risk_level": "low",
            "next_survival_pct": 0.0,
            "clear_routes": 0,
            "future_fits": 0,
            "sample_count": 0,
            "next_streak_pct": 0.0,
            "search_nodes": 0,
            "search_budget": 0,
            "search_budget_exhausted": False,
        }

    # Tahtayı bit maskesine dönüştürme
    board_mask = np.int64(0)
    for r in range(8):
        for c in range(8):
            if board[r, c] == 1:
                board_mask = np.int64(board_mask | np.int64(1) << np.int64(r * 8 + c))
    
    current_moves = np.full((3, 3), -1, dtype=np.int32)
    best_moves_global = np.full((3, 3), -1, dtype=np.int32)
    best_score_holder = np.array([-1e9], dtype=np.float64)
    best_diagnostics_holder = np.zeros(5, dtype=np.float64)
    effective_node_budget = config.SEARCH_NODE_BUDGET if node_budget is None else max(0, int(node_budget))
    nodes_visited_holder = np.zeros(1, dtype=np.int64)
    budget_exhausted_holder = np.zeros(1, dtype=np.bool_)

    # JIT arama motorunu çalıştır
    solve_recursive(
        board_mask,
        0,
        active_count,
        pieces_masks,
        pieces_shapes,
        pieces_active,
        current_moves,
        best_moves_global,
        best_score_holder,
        best_diagnostics_holder,
        0.0,
        config.W_CLEAR,
        config.W_EMPTY,
        config.W_HOLES,
        config.W_BUMPINESS,
        config.W_READINESS,
        config.W_FUTURE_FITS,
        config.W_LARGEST_REGION,
        config.W_SMALL_REGION_PENALTY,
        config.W_LINE_READINESS_SURVIVAL,
        config.W_TRAP_PENALTY,
        config.MONTE_CARLO_NORMAL_SAMPLES,
        config.MONTE_CARLO_DANGER_SAMPLES,
        config.MONTE_CARLO_DANGER_FILLED_CELLS,
        config.MONTE_CARLO_MIN_FUTURE_FITS,
        config.MONTE_CARLO_SEED,
        config.W_MONTE_CARLO_SURVIVAL,
        config.W_MONTE_CARLO_CLEAR_ROUTES,
        config.W_MONTE_CARLO_FUTURE_FITS,
        config.W_MONTE_CARLO_STREAK,
        config.W_STREAK_CONTINUE,
        config.W_STREAK_BREAK_PENALTY,
        config.W_STREAK_PERFECT_BONUS,
        0,  # initial streak_in_plan
        effective_node_budget,
        nodes_visited_holder,
        budget_exhausted_holder,
    )

    best_score = best_score_holder[0]
    diagnostics = _diagnostics_from_holder(best_score, best_diagnostics_holder)
    diagnostics.update({
        "search_nodes": int(nodes_visited_holder[0]),
        "search_budget": effective_node_budget,
        "search_budget_exhausted": bool(budget_exhausted_holder[0]),
    })

    # Hiçbir geçerli hamle bulunamadıysa (oyun bittiyse)
    if best_score < -1e8:
        return None, best_score, diagnostics

    # Çözüm hamlelerini listeye dök
    moves = []
    for d in range(active_count):
        idx = best_moves_global[d, 0]
        r = best_moves_global[d, 1]
        c = best_moves_global[d, 2]
        if idx != -1:
            moves.append({
                "slot_index": int(original_indices[idx]),
                "row": int(r),
                "col": int(c)
            })

    return moves, best_score, diagnostics


def solve(
    board: np.ndarray,
    active_pieces: List[Optional[np.ndarray]],
    node_budget: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], float]:
    moves, best_score, _ = _solve_core(board, active_pieces, node_budget)
    return moves, best_score


def solve_with_diagnostics(
    board: np.ndarray,
    active_pieces: List[Optional[np.ndarray]],
    node_budget: Optional[int] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], float, Dict[str, Any]]:
    return _solve_core(board, active_pieces, node_budget)
