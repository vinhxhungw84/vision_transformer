"""
data.py
=======
Bộ sinh ẢNH TỔNG HỢP cho ViT — KHÔNG cần tải dataset, KHÔNG thư viện ngoài.

Bài toán phân loại 4 lớp (mỗi ảnh xám 1 kênh, kích thước img_size×img_size):
    0 = sọc DỌC      (vertical stripes)
    1 = sọc NGANG    (horizontal stripes)
    2 = ô vuông sáng (blob) ở vị trí ngẫu nhiên
    3 = ô caro       (checkerboard)

Đây là các "kết cấu toàn cục" trải trên nhiều patch -> mô hình buộc phải GOM
thông tin qua token [CLS] (rất hợp để minh hoạ ViT).
"""
import numpy as np

CLASS_NAMES = ["sọc dọc", "sọc ngang", "ô vuông", "caro"]

def make_image(rng, size=16, cls=None):
    if cls is None:
        cls = rng.integers(0, 4)
    img = np.zeros((size, size), dtype=np.float64)
    period = 4
    if cls == 0:                              # sọc dọc
        cols = (np.arange(size) % period) < (period // 2)
        img[:, cols] = 1.0
    elif cls == 1:                            # sọc ngang
        rows = (np.arange(size) % period) < (period // 2)
        img[rows, :] = 1.0
    elif cls == 2:                            # ô vuông sáng ngẫu nhiên
        s = size // 3
        r = rng.integers(0, size - s + 1); c = rng.integers(0, size - s + 1)
        img[r:r+s, c:c+s] = 1.0
    else:                                     # caro
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        img = (((yy // period) + (xx // period)) % 2).astype(np.float64)
    # nhiễu nhẹ + chuẩn hoá về ~[-1,1]
    img = img + rng.normal(0, 0.1, size=img.shape)
    img = (img - 0.5) / 0.5
    return img, cls

def make_batch(rng, batch, size=16):
    """Trả về X: (B,1,H,W), y: (B,)."""
    X = np.zeros((batch, 1, size, size))
    y = np.zeros(batch, dtype=np.int64)
    for i in range(batch):
        img, cls = make_image(rng, size)
        X[i, 0] = img; y[i] = cls
    return X, y
