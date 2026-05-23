import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))); import config

# =====================================================================
# BLOCK BLAST SOLVER - GÖRSEL HUD VE ARTTIRILMIŞ GERÇEKLİK PANELİ (visualizer.py)
# =====================================================================

def summarize_move_sequence(board_state: np.ndarray,
                            pieces: List[Optional[np.ndarray]],
                            moves: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], int, int]:
    if not moves:
        return [], 0, int(np.sum(board_state))

    simulated_board = board_state.copy()
    outcomes = []
    total_clears = 0

    for move in moves:
        slot = move.get("slot_index", -1)
        row = move.get("row", -1)
        col = move.get("col", -1)
        cleared_rows = []
        cleared_cols = []
        invalid = False

        if slot < 0 or slot >= len(pieces) or pieces[slot] is None:
            invalid = True
        else:
            piece_matrix = pieces[slot]
            ph, pw = piece_matrix.shape

            for piece_r in range(ph):
                for piece_c in range(pw):
                    if piece_matrix[piece_r, piece_c] == 0:
                        continue

                    cell_row = row + piece_r
                    cell_col = col + piece_c
                    if cell_row < 0 or cell_row >= 8 or cell_col < 0 or cell_col >= 8:
                        invalid = True
                        continue
                    simulated_board[cell_row, cell_col] = 1

            if not invalid:
                for board_row in range(8):
                    if np.all(simulated_board[board_row, :] == 1):
                        cleared_rows.append(board_row)
                for board_col in range(8):
                    if np.all(simulated_board[:, board_col] == 1):
                        cleared_cols.append(board_col)

                if cleared_rows:
                    simulated_board[cleared_rows, :] = 0
                if cleared_cols:
                    simulated_board[:, cleared_cols] = 0

        clear_count = len(cleared_rows) + len(cleared_cols)
        total_clears += clear_count
        outcomes.append({
            "cleared_rows": cleared_rows,
            "cleared_cols": cleared_cols,
            "clear_count": clear_count,
            "filled_cells": int(np.sum(simulated_board)),
            "invalid": invalid,
        })

    return outcomes, total_clears, int(np.sum(simulated_board))

class Visualizer:
    """
    Kullanıcıya gösterilecek olan OpenCV penceresine görsel HUD
    öğelerini, kalibrasyon kutularını, dijital önizlemeyi ve hamle önerilerini çizer.
    """
    def __init__(self):
        self.step_colors = {
            1: (0, 0, 255),    # Adım 1: Kırmızı (BGR)
            2: (0, 255, 0),    # Adım 2: Yeşil
            3: (255, 0, 0)     # Adım 3: Mavi
        }

    def draw_hud(self,
                 frame: np.ndarray,
                 board_state: np.ndarray,
                 pieces: List[Optional[np.ndarray]],
                 moves: Optional[List[Dict[str, Any]]],
                 score: float,
                 occluded: bool,
                 diagnostics: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Kare üzerine tüm grafik overlay'leri (ROI sınırları, önizleme, yan panel, hamle kutuları) yerleştirir.
        """
        h, w, _ = frame.shape
        game_frame = frame.copy()
        panel_width = 320
        panel_x = w
        hud_frame = np.zeros((h, w + panel_width, 3), dtype=frame.dtype)
        hud_frame[:, :w] = game_frame

        # 1. Kalibrasyon Sınırlarını Çiz (BOARD_ROI ve PIECES_ROI)
        if config.BOARD_ROI is not None:
            bx = int(config.BOARD_ROI[0] * w)
            by = int(config.BOARD_ROI[1] * h)
            bw = int(config.BOARD_ROI[2] * w)
            bh = int(config.BOARD_ROI[3] * h)
            cv2.rectangle(hud_frame, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)  # Cyan
            cv2.putText(hud_frame, "BOARD ROI", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        if config.PIECES_ROI is not None:
            px = int(config.PIECES_ROI[0] * w)
            py = int(config.PIECES_ROI[1] * h)
            pw = int(config.PIECES_ROI[2] * w)
            ph = int(config.PIECES_ROI[3] * h)
            cv2.rectangle(hud_frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)  # Sarı
            cv2.putText(hud_frame, "PIECES ROI", (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Kontrol ve Bilgi Paneli Çiz (oyun görüntüsünün dışında)
        cv2.rectangle(hud_frame, (panel_x, 0), (panel_x + panel_width, h), (15, 15, 15), -1)

        # Sağ Panel Başlığı
        y_pos = 30
        cv2.putText(hud_frame, "BLOCK BLAST AI SOLVER", (panel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(hud_frame, (panel_x + 10, y_pos + 8), (panel_x + panel_width - 10, y_pos + 8), (100, 100, 100), 1)

        # Durum Bilgisi
        y_pos += 40
        if occluded:
            status_text = "ENGELENDI (Hand/Finger)"
            status_color = (0, 0, 255)  # Kırmızı
        else:
            status_text = "Aktif (Scanning)"
            status_color = (0, 255, 0)  # Yeşil

        cv2.putText(hud_frame, f"Durum: ", (panel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(hud_frame, status_text, (panel_x + 70, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        if diagnostics is not None:
            risk_level = str(diagnostics.get("risk_level", "unknown"))
            survival_pct = float(diagnostics.get("next_survival_pct", 0.0))
            clear_routes = int(diagnostics.get("clear_routes", 0))
            future_fits = int(diagnostics.get("future_fits", 0))
            sample_count = int(diagnostics.get("sample_count", 0))
            risk_color = (0, 255, 0)
            if risk_level == "medium":
                risk_color = (0, 200, 255)
            elif risk_level in ("high", "dead"):
                risk_color = (0, 0, 255)

            y_pos += 22
            cv2.putText(hud_frame, f"Risk: {risk_level.upper()}", (panel_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, risk_color, 1, cv2.LINE_AA)
            y_pos += 20
            cv2.putText(hud_frame, f"Next survival: {survival_pct:.0f}% ({sample_count})", (panel_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            y_pos += 18
            cv2.putText(hud_frame, f"Clear routes: {clear_routes} | Fits: {future_fits}", (panel_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            streak_pct = diagnostics.get("next_streak_pct")
            if streak_pct is not None:
                y_pos += 18
                streak_color = (0, 255, 255) if streak_pct >= 0.5 else (200, 200, 200)
                cv2.putText(hud_frame, f"Next streak: {int(streak_pct * 100)}%", (panel_x + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, streak_color, 1, cv2.LINE_AA)

        # 3. Küçük Dijital Tahta Önizlemesini Çiz (Yan Panelde)
        y_pos += 30
        self._draw_digital_board_preview(hud_frame, board_state, panel_x + 40, y_pos, size=15)

        # 4. Çözüm Önerilerini Çiz ve Yan Panele Yaz
        y_pos += 160
        cv2.putText(hud_frame, "HAMLE STRATEJISI:", (panel_x + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(hud_frame, (panel_x + 10, y_pos + 6), (panel_x + panel_width - 10, y_pos + 6), (100, 100, 100), 1)

        y_pos += 25
        if moves and config.BOARD_ROI is not None:
            move_outcomes, total_clears, final_filled = summarize_move_sequence(board_state, pieces, moves)
            bx = int(config.BOARD_ROI[0] * w)
            by = int(config.BOARD_ROI[1] * h)
            bw = int(config.BOARD_ROI[2] * w)
            bh = int(config.BOARD_ROI[3] * h)
            cell_w = bw / 8.0
            cell_h = bh / 8.0

            for idx, move in enumerate(moves):
                step_num = idx + 1
                slot = move["slot_index"]
                row = move["row"]
                col = move["col"]

                # İlgili adımdaki parça matrisini alarak kapladığı genişlik ve yüksekliği hesapla
                piece_matrix = pieces[slot]
                if piece_matrix is None:
                    continue
                ph, pw = piece_matrix.shape

                color = self.step_colors.get(step_num, (255, 255, 255))

                label_drawn = False
                for piece_r in range(ph):
                    for piece_c in range(pw):
                        if piece_matrix[piece_r, piece_c] == 0:
                            continue

                        cell_row = row + piece_r
                        cell_col = col + piece_c
                        if cell_row < 0 or cell_row >= 8 or cell_col < 0 or cell_col >= 8:
                            continue

                        px_start = bx + int(cell_col * cell_w)
                        py_start = by + int(cell_row * cell_h)
                        px_end = bx + int((cell_col + 1) * cell_w)
                        py_end = by + int((cell_row + 1) * cell_h)

                        cell_overlay = hud_frame.copy()
                        cv2.rectangle(cell_overlay, (px_start, py_start), (px_end, py_end), color, -1)
                        cv2.addWeighted(cell_overlay, 0.22, hud_frame, 0.78, 0, hud_frame)
                        cv2.rectangle(hud_frame, (px_start, py_start), (px_end, py_end), color, 3)

                        if not label_drawn:
                            txt = str(step_num)
                            cv2.putText(hud_frame, txt, (px_start + 10, py_start + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
                            cv2.putText(hud_frame, txt, (px_start + 10, py_start + 35),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
                            label_drawn = True

                # Yan panele metin olarak ekle
                move_text = f"{step_num}. P{slot + 1} -> ({row},{col})"
                if idx < len(move_outcomes):
                    clear_count = move_outcomes[idx]["clear_count"]
                    if clear_count > 0:
                        move_text += f" | +{clear_count} clear"
                cv2.putText(hud_frame, move_text, (panel_x + 10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                y_pos += 25
            
            cv2.putText(hud_frame, f"Toplam clear: {total_clears}", (panel_x + 10, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
            y_pos += 20
            cv2.putText(hud_frame, f"Kalan blok: {final_filled}", (panel_x + 10, y_pos + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
            y_pos += 20

            # AI Skor Değeri
            cv2.putText(hud_frame, f"Tahmini Skor: {score:.1f}", (panel_x + 10, y_pos + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            status_desc = "Cozum hesaplaniyor..."
            if occluded:
                status_desc = "Occluded (Goruntu Kapatildi)"
            elif not any(p is not None for p in pieces):
                status_desc = "Envanter Boş (Bekleniyor)"
            else:
                status_desc = "UYARI: Uygun hamle yok!"
            
            cv2.putText(hud_frame, status_desc, (panel_x + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if "UYARI" in status_desc else (150, 150, 150), 1, cv2.LINE_AA)

        # Kısayol Tuşları Bilgilendirmesi (En Alt)
        y_pos_controls = h - 60
        cv2.putText(hud_frame, "KONTROLLER:", (panel_x + 10, y_pos_controls),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud_frame, "c: Yeni Kalibrasyon", (panel_x + 10, y_pos_controls + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(hud_frame, "q: Programdan Cikis", (panel_x + 10, y_pos_controls + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        return hud_frame

    def _draw_digital_board_preview(self, frame: np.ndarray, board_state: np.ndarray, x: int, y: int, size: int):
        """
        8x8 dijital tahta durumunu yan panele küçük renkli kutucuklar halinde çizer.
        """
        for r in range(8):
            for c in range(8):
                # Dolu hücreler açık gri/mavi, boş hücreler koyu gri
                color = (180, 180, 180) if board_state[r, c] == 1 else (45, 45, 45)
                # Kutucuğun konumunu hesapla
                px1 = x + c * (size + 2)
                py1 = y + r * (size + 2)
                px2 = px1 + size
                py2 = py1 + size
                cv2.rectangle(frame, (px1, py1), (px2, py2), color, -1)
                # Çerçeve
                cv2.rectangle(frame, (px1, py1), (px2, py2), (80, 80, 80), 1)
