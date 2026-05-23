import cv2
import time
import numpy as np
import config
from modules.capture import WindowCapture
from modules.visualizer import Visualizer
import modules.vision as vision
import modules.solver as solver

# =====================================================================
# BLOCK BLAST SOLVER - ANA ORKESTRASYON VE PROGRAM DÖNGÜSÜ (main.py)
# =====================================================================

def check_state_changed(board1: np.ndarray, board2: np.ndarray,
                        pieces1: list, pieces2: list) -> bool:
    """
    Tahta durumunun veya inventory'deki parçaların değişip değişmediğini kontrol eder.
    Gereksiz çözücü (solver) çağrılarını engellemek ve işlemci yükünü azaltmak için kullanılır.
    """
    # Tahta durumunu karşılaştır
    if not np.array_equal(board1, board2):
        return True

    # Parçaları karşılaştır (sayı, varlık/yokluk ve şekil olarak)
    if len(pieces1) != len(pieces2):
        return True

    for i in range(len(pieces1)):
        p1 = pieces1[i]
        p2 = pieces2[i]
        # Biri boşken diğeri doluysa durum değişmiştir
        if (p1 is None) != (p2 is None):
            return True
        # İkisi de doluysa içeriklerini karşılaştır
        if p1 is not None and p2 is not None:
            if not np.array_equal(p1, p2):
                return True

    return False


def copy_pieces(pieces: list) -> list:
    return [piece.copy() if piece is not None else None for piece in pieces]


def main():
    print("[BİLGİ] Block Blast AI Solver Başlatılıyor...")
    
    # 1. Kalibrasyon Bilgilerini Yükle
    config.load_calibration()

    # 2. Yakalama, Görselleştirici ve Pencere Kontrollerini Başlat
    capture = WindowCapture(config.WINDOW_TITLE)
    visualizer = Visualizer()

    # Eğer kalibrasyon verisi yoksa veya geçersizse kullanıcıyı kalibrasyona yönlendir
    if config.BOARD_ROI is None or config.PIECES_ROI is None:
        print("[UYARI] Kalibrasyon verisi eksik. Kalibrasyon ekranı açılıyor...")
        if not capture.calibrate():
            print("[HATA] Kalibrasyon tamamlanamadı. Program sonlandırılıyor.")
            return

    # Durum takibi için önbellek değişkenleri
    last_board_state = np.zeros((8, 8), dtype=np.uint8)
    last_pieces = [None, None, None]
    last_moves = None
    last_score = 0.0
    last_diagnostics = None
    pending_board_state = None
    pending_pieces = [None, None, None]
    stable_frame_count = 0

    print("[BİLGİ] AI Solver ana döngüsü aktif. Çıkmak için HUD penceresi açıkken 'q' tuşuna basın.")
    
    cv2.namedWindow("Block Blast AI Solver HUD", cv2.WINDOW_NORMAL)

    while True:
        loop_start_time = time.time()

        # Ekran görüntüsünü yakala
        frame = capture.capture_frame()
        if frame is None:
            # Pencere bulunamadıysa veya minimize ise bekle ve tekrar aramayı tetikle
            time.sleep(0.1)
            continue

        # Hücre boyutlarını piksel koordinatları cinsinden hesapla (envanter tespiti için gerekli)
        bx, by, bw, bh = capture.get_roi_rect(config.BOARD_ROI, frame.shape)
        cell_w = bw / 8.0
        cell_h = bh / 8.0

        # Tahtanın durumunu oku ve el kapaması (occlusion) olup olmadığını incele
        detected_board_state, occluded = vision.get_board_state(frame)

        # Envanter parçalarını tespit et
        detected_pieces = vision.get_pieces(frame, cell_w, cell_h)
        board_state = last_board_state.copy()
        pieces = copy_pieces(last_pieces)

        if occluded:
            # Görüntü kapatıldıysa (örn: sürükleme anında el kapatması) çözücüyü atla
            # Ancak son bilinen geçerli tavsiyeleri HUD üzerinde göstermeye devam et
            print("[UYARI: Kare Engellendi / Frame Occluded] Solver atlanıyor...")
            stable_frame_count = 0
        else:
            if pending_board_state is not None and not check_state_changed(
                detected_board_state, pending_board_state, detected_pieces, pending_pieces
            ):
                stable_frame_count += 1
            else:
                pending_board_state = detected_board_state.copy()
                pending_pieces = copy_pieces(detected_pieces)
                stable_frame_count = 1

            if stable_frame_count < config.DETECTION_STABLE_FRAMES:
                hud_frame = visualizer.draw_hud(frame, board_state, pieces, last_moves, last_score, occluded, last_diagnostics)
                cv2.imshow("Block Blast AI Solver HUD", hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[BİLGİ] Çıkış yapılıyor...")
                    break
                elif key == ord('c'):
                    cv2.destroyWindow("Block Blast AI Solver HUD")
                    capture.calibrate()
                    cv2.namedWindow("Block Blast AI Solver HUD", cv2.WINDOW_NORMAL)
                    last_board_state = np.zeros((8, 8), dtype=np.uint8)
                    last_pieces = [None, None, None]
                    last_moves = None
                    last_score = 0.0
                    last_diagnostics = None
                    pending_board_state = None
                    pending_pieces = [None, None, None]
                    stable_frame_count = 0
                continue

            board_state = pending_board_state.copy()
            pieces = copy_pieces(pending_pieces)

            # Durum değişti mi kontrol et (Yeni hamle yapılmış mı veya parça yerleştirilmiş mi?)
            state_changed = check_state_changed(board_state, last_board_state, pieces, last_pieces)

            if state_changed:
                print("[BİLGİ] Durum değişikliği algılandı. Yeni hamle hesaplanıyor...")
                
                # Envanterde aktif parça var mı bak
                has_active_pieces = any(p is not None for p in pieces)

                if has_active_pieces:
                    solver_start = time.time()
                    
                    # Numba hızlandırılmış backtracking çözücüyü çalıştır
                    moves, score, diagnostics = solver.solve_with_diagnostics(board_state, pieces)
                    
                    solver_duration = (time.time() - solver_start) * 1000.0
                    print(f"[BİLGİ] Çözüm hesaplandı. Süre: {solver_duration:.1f}ms | Skor: {score:.1f}")

                    # Önbelleği güncelle
                    last_moves = moves
                    last_score = score
                    last_diagnostics = diagnostics
                else:
                    # Envanter tamamen boşsa (yeni parça gelmesi bekleniyor)
                    last_moves = None
                    last_score = 0.0
                    last_diagnostics = None

                # Son geçerli durumları kaydet
                last_board_state = board_state.copy()
                last_pieces = copy_pieces(pieces)

        # 3. HUD Görsellerini Kareye Çiz ve Ekranda Göster
        hud_frame = visualizer.draw_hud(frame, board_state, pieces, last_moves, last_score, occluded, last_diagnostics)
        cv2.imshow("Block Blast AI Solver HUD", hud_frame)

        # Pencere boyutunu makul bir seviyeye getirelim (opsiyonel)
        # cv2.resizeWindow("Block Blast AI Solver HUD", 800, 600)

        # Klavye Kontrollerini Dinle
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Programdan çıkış
            print("[BİLGİ] Çıkış yapılıyor...")
            break
        elif key == ord('c'):
            # Zorla yeniden kalibrasyon yap
            cv2.destroyWindow("Block Blast AI Solver HUD")
            capture.calibrate()
            cv2.namedWindow("Block Blast AI Solver HUD", cv2.WINDOW_NORMAL)
            # Durum önbelleğini sıfırla
            last_board_state = np.zeros((8, 8), dtype=np.uint8)
            last_pieces = [None, None, None]
            last_moves = None
            last_score = 0.0
            last_diagnostics = None
            pending_board_state = None
            pending_pieces = [None, None, None]
            stable_frame_count = 0

        # Latency hedefi doğrulaması (250ms altı kontrolü)
        loop_duration = (time.time() - loop_start_time) * 1000.0
        # print(f"[DEBUG] Toplam döngü süresi: {loop_duration:.1f}ms")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
