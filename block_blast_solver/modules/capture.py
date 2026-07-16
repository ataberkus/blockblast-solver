import ctypes
import logging
import sys
import time
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

from block_blast_solver import config

logger = logging.getLogger(__name__)


class CaptureBackend(Protocol):
    def capture_frame(self) -> Optional[np.ndarray]: ...

    def calibrate(self) -> bool: ...

    def get_roi_rect(
        self,
        roi_ratios: list,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]: ...

    def close(self) -> None: ...


def roi_to_rect(
    roi_ratios: list,
    frame_shape: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    valid, reason = config.validate_roi(roi_ratios)
    if not valid:
        raise ValueError(reason)
    height, width, _ = frame_shape
    x, y, roi_width, roi_height = roi_ratios
    return (
        int(x * width),
        int(y * height),
        int(roi_width * width),
        int(roi_height * height),
    )


class StaticFrameCapture:
    """Headless capture backend for replay tests and local vision debugging."""

    def __init__(self, frame: np.ndarray):
        self.frame = frame.copy()

    def capture_frame(self) -> Optional[np.ndarray]:
        return self.frame.copy()

    def calibrate(self) -> bool:
        return config.BOARD_ROI is not None and config.PIECES_ROI is not None

    def get_roi_rect(
        self,
        roi_ratios: list,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        return roi_to_rect(roi_ratios, frame_shape)

    def close(self) -> None:
        return None

# =====================================================================
# BLOCK BLAST SOLVER - EKRAN YAKALAMA VE KALİBRASYON MODÜLÜ (capture.py)
# =====================================================================

class WindowCapture:
    """
    Hedef LonelyScreen penceresini yakalamaktan, ekran görüntüsünü düşük gecikmeyle
    çekmekten ve kullanıcıdan ROI kalibrasyonu almaktan sorumlu sınıftır.
    """
    def __init__(self, window_title: str):
        if sys.platform != "win32":
            raise RuntimeError("Live window capture requires Windows; use StaticFrameCapture for headless runs")
        self.window_title = window_title
        self.cameras = {}       # output_idx -> dxcam camera
        self.monitor_rects = {} # output_idx -> (left, top, right, bottom)
        self._init_cameras()
        self.window_handle = None
        self.last_bbox = None
        self.find_window()

    def _init_cameras(self):
        """Enumerate all monitors and create a dxcam camera for each."""
        import dxcam

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
                logger.info("Display %d detected: %dx%d @ (%d,%d)", idx, mr - ml, mb - mt, ml, mt)
            except Exception as error:
                logger.warning("Could not create camera for display %d: %s", idx, error)

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
        import pygetwindow as gw

        windows = gw.getWindowsWithTitle(self.window_title)
        if not windows:
            logger.warning("Window not found: %s", self.window_title)
            self.window_handle = None
            return False

        self.window_handle = windows[0]
        
        # Simge durumuna küçültülmüşse pencereyi eski haline getir
        if self.window_handle.isMinimized:
            try:
                self.window_handle.restore()
                logger.info("Restored minimized window: %s", self.window_title)
            except Exception as error:
                logger.warning("Could not restore window: %s", error)

        # Pencereyi aktif hale getirmeyi deneyelim (opsiyonel)
        try:
            self.window_handle.activate()
        except Exception:
            # Bazı Windows güvenlik ayarları doğrudan aktivasyona izin vermeyebilir
            pass

        logger.info(
            "Window found: %s at (%d, %d), size %dx%d",
            self.window_handle.title,
            self.window_handle.left,
            self.window_handle.top,
            self.window_handle.width,
            self.window_handle.height,
        )
        return True

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture the target window through DXGI and return a BGR numpy array.
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

        except Exception as error:
            logger.error("Screen capture failed: %s", error)
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
        logger.info("Starting calibration; ensure the target window is visible")

        # Pencere ekran dışındaysa kullanıcıyı uyar ve pencereyi taşı
        frame = self.capture_frame()
        if frame is None:
            if self.window_handle is not None:
                logger.warning("Window appears off-screen; moving it to the primary display")
                try:
                    mon_idx = list(self.cameras.keys())[0]
                    ml, mt, mr, mb = self.monitor_rects[mon_idx]
                    self.window_handle.moveTo((ml + mr) // 4, (mt + mb) // 4)
                    time.sleep(0.5)
                    frame = self.capture_frame()
                except Exception as error:
                    logger.error("Could not move window: %s", error)
            if frame is None:
                logger.error("Could not capture a frame for calibration")
                return False

        h, w, _ = frame.shape
        logger.info("Captured %dx%d frame; select the game board", w, h)

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
            logger.warning("Board selection cancelled")
            return False

        # 2. Alt Parçaların Bölgesini Seç (Kullanıcıya açıklayıcı metin ekleyelim)
        logger.info("Select the inventory area containing all three pieces")
        pieces_instruction_frame = frame.copy()
        cv2.putText(pieces_instruction_frame, "SADECE ALTTAN 3 PARCA ALANINI SECIN!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.namedWindow("Kalibrasyon - Parca Alanini Secin", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Kalibrasyon - Parca Alanini Secin", cv2.WND_PROP_TOPMOST, 1)
        r_pieces = cv2.selectROI("Kalibrasyon - Parca Alanini Secin", pieces_instruction_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Kalibrasyon - Parca Alanini Secin")

        if r_pieces[2] == 0 or r_pieces[3] == 0:
            logger.warning("Inventory selection cancelled")
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
        return config.save_calibration(board_roi, pieces_roi)

    def get_roi_rect(self, roi_ratios: list, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Oransal koordinatları (ratios) verilen kare boyutuna göre piksel koordinatlarına (x, y, w, h) geri dönüştürür.
        """
        return roi_to_rect(roi_ratios, frame_shape)

    def close(self) -> None:
        for output_idx, camera in list(self.cameras.items()):
            for method_name in ("stop", "release"):
                method = getattr(camera, method_name, None)
                if not callable(method):
                    continue
                try:
                    method()
                except Exception as error:
                    logger.debug("Could not %s dxcam camera %s: %s", method_name, output_idx, error)
        self.cameras.clear()
        self.monitor_rects.clear()
        self.window_handle = None
