import os
import json

# =====================================================================
# BLOCK BLAST SOLVER - KONFİGÜRASYON VE PAYLAŞILAN DURUM MODÜLÜ (config.py)
# =====================================================================

# Pencere başlığı ve kalibrasyon dosya yolu tanımları
WINDOW_TITLE = "LonelyScreen AirPlay Receiver"
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_data.json")

# Hücre tespiti için doku varyansı eşik değeri (Tile Variance Threshold)
# Ahşap, mücevher vb. temalarda hücrelerin dolu/boş ayrımını yapmak için kullanılır.
TILE_VARIANCE_THRESHOLD = 8.5

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
W_MONTE_CARLO_SURVIVAL = 120.0
W_MONTE_CARLO_CLEAR_ROUTES = 18.0
W_MONTE_CARLO_FUTURE_FITS = 12.0

# Streak / combo continuity weights (Section 1 + Section 2 of streak spec)
W_STREAK_CONTINUE        =  250.0   # per placement in current plan that clears >= 1 line
W_STREAK_BREAK_PENALTY   = -900.0   # per placement in current plan that clears 0 lines
W_STREAK_PERFECT_BONUS   =  600.0   # one-shot bonus when all 3 placements clear
W_MONTE_CARLO_STREAK     =  200.0   # scaled by avg_streak_continuity in [0.0, 1.0]

# Kalibrasyon ROI (Region of Interest) Değişkenleri
# Pencere boyutuna oranlanmış şekilde (normalized ratios: 0.0 - 1.0) saklanır.
BOARD_ROI = None   # 8x8 Grid alanı [x_start_ratio, y_start_ratio, width_ratio, height_ratio]
PIECES_ROI = None  # Alt kısımdaki 3 parça alanı [x_start_ratio, y_start_ratio, width_ratio, height_ratio]


def save_calibration(board_roi, pieces_roi):
    """
    Kullanıcı tarafından seçilen ROI koordinat oranlarını JSON formatında diske kaydeder.
    Bu sayede uygulama her yeniden başlatıldığında kalibrasyon adımı atlanabilir.
    """
    global BOARD_ROI, PIECES_ROI
    BOARD_ROI = board_roi
    PIECES_ROI = pieces_roi

    data = {
        "BOARD_ROI": board_roi,
        "PIECES_ROI": pieces_roi
    }
    try:
        with open(CALIBRATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"[BİLGİ] Kalibrasyon verileri başarıyla kaydedildi: {CALIBRATION_FILE}")
    except Exception as e:
        print(f"[HATA] Kalibrasyon verileri kaydedilemedi: {e}")


def load_calibration():
    """
    Diskteki JSON kalibrasyon dosyasını okur ve ROI koordinat oranlarını yükler.
    Dosya bulunamazsa veya hata oluşursa None döndürür.
    """
    global BOARD_ROI, PIECES_ROI
    if not os.path.exists(CALIBRATION_FILE):
        print("[UYARI] Kalibrasyon dosyası bulunamadı. Yeni bir kalibrasyon gerekmektedir.")
        return False

    try:
        with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        BOARD_ROI = data.get("BOARD_ROI")
        PIECES_ROI = data.get("PIECES_ROI")
        print("[BİLGİ] Kalibrasyon verileri başarıyla yüklendi.")
        return True
    except Exception as e:
        print(f"[HATA] Kalibrasyon verileri yüklenirken hata oluştu: {e}")
        return False
