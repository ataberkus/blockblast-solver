import cv2
import ctypes
import dxcam
import numpy as np
import pygetwindow as gw
from typing import Tuple, Optional
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))); import config

# =====================================================================
# BLOCK BLAST SOLVER - EKRAN YAKALAMA VE KALİBRASYON MODÜLÜ (capture.py)
# =====================================================================

class WindowCapture:
    """
    Hedef pencereyi (örn: UxPlay) yakalamaktan, ekran görüntüsünü düşük gecikmeyle
    çekmekten ve kullanıcıdan ROI kalibrasyonu almaktan sorumlu sınıftır.
    """
    def __init__(self, window_title: str):
        self.window_title = window_title
        self.cameras = {}       # output_idx -> dxcam camera
        self.monitor_rects = {} # output_idx -> (left, top, right, bottom)
        self._init_cameras()
        self.window_handle = None
        self.last_bbox = None
        self.find_window()

    def _init_cameras(self):
        """Enumerate all monitors and create a dxcam camera for each."""
        class RECT(ctypes.Structure):
            _fields_ = [('left', ctypes.c_long), ('top', ctypes.c_long),
                        ('right', ctypes.c_long), ('bottom', ctypes.c_long)]

        monitors = []
        MonitorEnumProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.POINTER(RECT), ctypes.c_double
        )

        def _cb(hmon, hdc, rect, data):
            r = rect.contents
            monitors.append((r.left, r.top, r.right, r.bottom))
            return True

        ctypes.windll.user32.EnumDisplayMonitors(None, None, MonitorEnumProc(_cb), 0)

        for idx, rect in enumerate(monitors):
            try:
                cam = dxcam.create(output_idx=idx, output_color="BGR")
                self.cameras[idx] = cam
                self.monitor_rects[idx] = rect
                ml, mt, mr, mb = rect
                print(f"[BİLGİ] Ekran {idx} tespit edildi: {mr - ml}x{mb - mt} @ ({ml},{mt})")
            except Exception as e:
                print(f"[UYARI] Ekran {idx} kamerası oluşturulamadı: {e}")

        if not self.cameras:
            cam = dxcam.create(output_color="BGR")
            self.cameras[0] = cam
            self.monitor_rects[0] = (0, 0, cam.width, cam.height)

    def _find_monitor_idx(self, win_left, win_top, win_right, win_bottom) -> int:
        """Return the output_idx of the monitor containing the most window area."""
        best_idx = list(self.cameras.keys())[0]
        best_overlap = 0
        for idx, (ml, mt, mr, mb) in self.monitor_rects.items():
            ol = max(0, min(win_right, mr) - max(win_left, ml))
            oh = max(0, min(win_bottom, mb) - max(win_top, mt))
            overlap = ol * oh
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
        return best_idx

    def find_window(self) -> bool:
        """
        pygetwindow kütüphanesi kullanarak hedef pencereyi bulur.
        Eğer pencere simge durumuna küçültülmüşse (minimized) geri yükler (restore).
        """
        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            print(f"[UYARI] '{self.window_title}' başlığına sahip bir pencere bulunamadı.")
            self.window_handle = None
            return False

        self.window_handle = windows[0]
        
        # Simge durumuna küçültülmüşse pencereyi eski haline getir
        if self.window_handle.isMinimized:
            try:
                self.window_handle.restore()
                print(f"[BİLGİ] '{self.window_title}' penceresi simge durumundan kurtarıldı.")
            except Exception as e:
                print(f"[UYARI] Pencere kurtarılamadı: {e}")

        # Pencereyi aktif hale getirmeyi deneyelim (opsiyonel)
        try:
            self.window_handle.activate()
        except Exception:
            # Bazı Windows güvenlik ayarları doğrudan aktivasyona izin vermeyebilir
            pass

        print(f"[BİLGİ] Pencere bulundu: {self.window_handle.title} | "
              f"Konum: ({self.window_handle.left}, {self.window_handle.top}) | "
              f"Boyut: {self.window_handle.width}x{self.window_handle.height}")
        return True

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        mss kütüphanesi kullanarak hedef pencerenin sınırlarını yüksek hızda yakalar.
        Renk uzayını BGRA'dan standard BGR'a dönüştürür ve numpy array olarak döndürür.
        """
        if self.window_handle is None or not self._is_window_valid(self.window_handle):
            if not self.find_window():
                return None

        try:
            # Pencerenin anlık koordinatlarını al
            left = self.window_handle.left
            top = self.window_handle.top
            width = self.window_handle.width
            height = self.window_handle.height

            # Geçersiz boyut kontrolü (örn. pencere kapanırken ya da minimize iken sıfır olabilir)
            if width <= 0 or height <= 0:
                return None

            # Hangi monitörde olduğunu bul ve koordinatları o monitöre göre ayarla
            mon_idx = self._find_monitor_idx(left, top, left + width, top + height)
            camera = self.cameras[mon_idx]
            ml, mt, mr, mb = self.monitor_rects[mon_idx]

            r_left   = max(0, left - ml)
            r_top    = max(0, top - mt)
            r_right  = min(left + width - ml, mr - ml)
            r_bottom = min(top + height - mt, mb - mt)

            # Görünür alan çok küçükse atla
            if r_right - r_left < 10 or r_bottom - r_top < 10:
                return None

            region = (r_left, r_top, r_right, r_bottom)
            self.last_bbox = region

            # DXGI Desktop Duplication ile yakala (donanım hızlandırmalı pencereler dahil)
            frame_bgr = camera.grab(region=region)
            if frame_bgr is None:
                return None
            return frame_bgr

        except Exception as e:
            print(f"[HATA] Ekran görüntüsü yakalanamadı: {e}")
            self.window_handle = None  # Bir sonraki döngüde yeniden aramayı tetiklesin diye sıfırlıyoruz
            return None

    def _is_window_valid(self, win) -> bool:
        """
        Pencere tutamacının hala geçerli olup olmadığını kontrol eder.
        """
        try:
            # left özelliğine erişmek pencerenin varlığını doğrulamak için basit bir testtir
            _ = win.left
            return True
        except Exception:
            return False

    def calibrate(self) -> bool:
        """
        Ekran görüntüsünden bir kare alır ve kullanıcıya iki bölge seçtirir:
        1. 8x8 Oyun Alanı (BOARD_ROI)
        2. Alt Parça Alanı (PIECES_ROI)
        Seçilen bölgeleri normalleştirilmiş oranlar halinde calibration_data.json dosyasına kaydeder.
        """
        print("[BİLGİ] Kalibrasyon başlatılıyor. Lütfen pencerenin görünür olduğundan emin olun...")

        # Pencere ekran dışındaysa kullanıcıyı uyar ve pencereyi taşı
        frame = self.capture_frame()
        if frame is None:
            if self.window_handle is not None:
                print("[UYARI] Pencere ekran dışında görünüyor. Pencere ekranın ortasına taşınıyor...")
                try:
                    mon_idx = list(self.cameras.keys())[0]
                    ml, mt, mr, mb = self.monitor_rects[mon_idx]
                    self.window_handle.moveTo((ml + mr) // 4, (mt + mb) // 4)
                    import time; time.sleep(0.5)
                    frame = self.capture_frame()
                except Exception as e:
                    print(f"[HATA] Pencere taşınamadı: {e}")
            if frame is None:
                print("[HATA] Kalibrasyon için ekran görüntüsü alınamadı. LonelyScreen penceresini ekrana taşıyın ve tekrar deneyin.")
                return False

        h, w, _ = frame.shape
        print(f"[BİLGİ] Ekran görüntüsü alındı: {w}x{h}. Lütfen oyun alanını seçin.")

        # 1. 8x8 Ana Oyun Tahtasını Seç (Kullanıcıya açıklayıcı metin ekleyelim)
        board_instruction_frame = frame.copy()
        # Üzerine açıklama yazısı çizelim
        cv2.putText(board_instruction_frame, "SADECE 8x8 OYUN TAHTASINI SECIN!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(board_instruction_frame, "(Alttaki 3 parcayi dahil ETMEYIN)", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.namedWindow("Kalibrasyon - Oyun Tahtasini Secin", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Kalibrasyon - Oyun Tahtasini Secin", cv2.WND_PROP_TOPMOST, 1)
        r_board = cv2.selectROI("Kalibrasyon - Oyun Tahtasini Secin", board_instruction_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Kalibrasyon - Oyun Tahtasini Secin")

        # Geçersiz ROI kontrolü (seçim iptal edilirse w veya h sıfır olur)
        if r_board[2] == 0 or r_board[3] == 0:
            print("[UYARI] Oyun tahtası seçimi iptal edildi.")
            return False

        # 2. Alt Parçaların Bölgesini Seç (Kullanıcıya açıklayıcı metin ekleyelim)
        print("[BİLGİ] Lütfen aşağıdaki 3 parçayı kapsayan envanter alanını seçin.")
        pieces_instruction_frame = frame.copy()
        cv2.putText(pieces_instruction_frame, "SADECE ALTTAN 3 PARCA ALANINI SECIN!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.namedWindow("Kalibrasyon - Parca Alanini Secin", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Kalibrasyon - Parca Alanini Secin", cv2.WND_PROP_TOPMOST, 1)
        r_pieces = cv2.selectROI("Kalibrasyon - Parca Alanini Secin", pieces_instruction_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Kalibrasyon - Parca Alanini Secin")

        if r_pieces[2] == 0 or r_pieces[3] == 0:
            print("[UYARI] Parça alanı seçimi iptal edildi.")
            return False

        # Koordinatları pencere genişliği ve yüksekliğine bölerek oranla (normalleştir)
        # Format: [x_ratio, y_ratio, w_ratio, h_ratio]
        board_roi = [
            float(r_board[0] / w),
            float(r_board[1] / h),
            float(r_board[2] / w),
            float(r_board[3] / h)
        ]

        pieces_roi = [
            float(r_pieces[0] / w),
            float(r_pieces[1] / h),
            float(r_pieces[2] / w),
            float(r_pieces[3] / h)
        ]

        # Konfigürasyona ve JSON dosyasına kaydet
        config.save_calibration(board_roi, pieces_roi)
        return True

    def get_roi_rect(self, roi_ratios: list, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Oransal koordinatları (ratios) verilen kare boyutuna göre piksel koordinatlarına (x, y, w, h) geri dönüştürür.
        """
        h, w, _ = frame_shape
        rx, ry, rw, rh = roi_ratios
        return (
            int(rx * w),
            int(ry * h),
            int(rw * w),
            int(rh * h)
        )
