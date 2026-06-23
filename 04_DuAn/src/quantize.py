"""
quantize.py  (PHẦN MỞ RỘNG — liên hệ FPGA)
==========================================
Lượng tử hoá trọng số ViT sang Q8.8 (int16, 8 bit phân số) rồi đo sụt chính xác.
Giống quy trình so float vs Q8.8 trong dự án SRCNN của bạn.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from vit import ViT, Config, cross_entropy, Adam
from data import make_batch
from train import IMG, PATCH, evaluate

FRAC = 8; SCALE = 1 << FRAC
QMIN, QMAX = -(1 << 15), (1 << 15) - 1

def to_q88(x):
    return np.clip(np.round(x * SCALE), QMIN, QMAX) / SCALE

def quantize_model(model):
    for owner, pname, _ in model.params():
        p = getattr(owner, pname); p[...] = to_q88(p)

def train_quick(steps=1000):
    rng = np.random.default_rng(0)
    cfg = Config(img_size=IMG, patch_size=PATCH, num_classes=4,
                 n_layer=3, n_head=4, n_embd=64, seed=1337)
    model = ViT(cfg)
    opt = Adam(model.params(), lr=3e-3, betas=(0.9, 0.95), weight_decay=1e-4)
    for _ in range(steps):
        X, y = make_batch(rng, 64, IMG)
        _, dl = cross_entropy(model.forward(X), y)
        model.backward(dl); opt.step()
    return model

def main():
    print("Huấn luyện nhanh mô hình float...")
    model = train_quick()
    acc_f = evaluate(model, np.random.default_rng(123))
    print(f"Độ chính xác (float32) : {acc_f*100:.2f}%")

    allw = np.concatenate([getattr(o, pn).ravel() for o, pn, _ in model.params()])
    print(f"Trọng số: min={allw.min():.3f} max={allw.max():.3f} std={allw.std():.3f}")

    print("\nLượng tử hoá sang Q8.8...")
    quantize_model(model)
    acc_q = evaluate(model, np.random.default_rng(123))
    print(f"Độ chính xác (Q8.8)    : {acc_q*100:.2f}%")
    print(f"Sụt độ chính xác       : {(acc_f-acc_q)*100:.2f} điểm %")
    print("\n=> Phần cần xử lý kỹ trên FPGA: softmax & LayerNorm (exp, chia, căn).")

if __name__ == "__main__":
    main()
