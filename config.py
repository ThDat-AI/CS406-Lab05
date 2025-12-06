# config.py
import os

# Đường dẫn đến model
MODEL_PATH = os.path.join("model", "best.pt")

# Ngưỡng tin cậy (Confidence Threshold) mặc định
CONFIDENCE_THRESHOLD = 0.50

# Định nghĩa các lớp (Classes) theo thứ tự ID của model (0, 1, 2)
CLASS_NAMES = {
    0: 'with_mask',
    1: 'mask_weared_incorrect',
    2: 'without_mask'
}

# Định nghĩa màu sắc cho từng lớp (R, G, B)
# with_mask -> Xanh lá (Green)
# without_mask -> Đỏ (Red)
# mask_weared_incorrect -> Vàng (Yellow)
COLORS = {
    0: (0, 255, 0),    # Green
    1: (255, 255, 0),  # Yellow
    2: (255, 0, 0)     # Red
}