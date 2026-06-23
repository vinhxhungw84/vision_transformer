"""
train.py
========
Huấn luyện Vision Transformer (tự cài bằng NumPy) phân loại ảnh tổng hợp 4 lớp:
sọc dọc / sọc ngang / ô vuông / caro.

Chạy:  PYTHONUTF8=1 python train.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from vit import ViT, Config, cross_entropy, Adam
from data import make_batch, CLASS_NAMES

IMG = 16          # kích thước ảnh
PATCH = 4         # -> 4x4 = 16 patch
BATCH = 64
STEPS = 600        # hội tụ rất sớm (~bước 150); 600 là dư
LR = 3e-3

def evaluate(model, rng, n=500):
    X, y = make_batch(rng, n, IMG)
    logits = model.forward(X)
    pred = logits.argmax(axis=1)
    return (pred == y).mean()

def main():
    rng = np.random.default_rng(0)
    cfg = Config(img_size=IMG, patch_size=PATCH, num_classes=4,
                 n_layer=3, n_head=4, n_embd=64, seed=1337)
    model = ViT(cfg)
    opt = Adam(model.params(), lr=LR, betas=(0.9, 0.95), weight_decay=1e-4)

    nparam = sum(getattr(o, pn).size for o, pn, _ in model.params())
    print(f"Tham số mô hình: {nparam:,}")
    print(f"Ảnh {IMG}x{IMG}, patch {PATCH}x{PATCH} -> {cfg.n_patch} patch + 1 [CLS]\n")

    for step in range(1, STEPS+1):
        X, y = make_batch(rng, BATCH, IMG)
        logits = model.forward(X)
        loss, dlogits = cross_entropy(logits, y)
        model.backward(dlogits)
        opt.step()
        if step % 150 == 0 or step == 1:
            acc = evaluate(model, np.random.default_rng(123))
            print(f"bước {step:5d} | loss {loss:.4f} | độ chính xác {acc*100:5.1f}%")

    # vài ví dụ
    print("\n--- Ví dụ dự đoán ---")
    rng2 = np.random.default_rng(7)
    X, y = make_batch(rng2, 8, IMG)
    pred = model.forward(X).argmax(axis=1)
    for i in range(8):
        ok = "✅" if pred[i] == y[i] else "❌"
        print(f"thật: {CLASS_NAMES[y[i]]:9s} | đoán: {CLASS_NAMES[pred[i]]:9s} {ok}")

if __name__ == "__main__":
    main()
