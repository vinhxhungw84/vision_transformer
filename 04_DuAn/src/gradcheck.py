"""
gradcheck.py
============
Kiểm tra backprop của ViT bằng GRADIENT SỐ:
    (L(theta+eps) - L(theta-eps)) / (2*eps)  ~  gradient giải tích
Sai số tương đối ~1e-5 => backward đã đúng.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from vit import ViT, Config, cross_entropy

def run():
    cfg = Config(img_size=8, patch_size=4, num_classes=4,
                 n_layer=2, n_head=2, n_embd=16, seed=0)
    model = ViT(cfg)
    rng = np.random.default_rng(42)
    B = 4
    X = rng.normal(0, 1, size=(B, 1, 8, 8))
    y = rng.integers(0, 4, size=B)

    def loss_fn():
        logits = model.forward(X)
        loss, dlogits = cross_entropy(logits, y)
        return loss, dlogits

    loss, dlogits = loss_fn()
    model.backward(dlogits)

    eps = 1e-5; max_rel = 0.0; checked = 0
    for owner, pname, gname in model.params():
        p = getattr(owner, pname); g = getattr(owner, gname)
        flat = p.ravel(); gflat = g.ravel()
        for k in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            orig = flat[k]
            flat[k] = orig + eps; lp, _ = loss_fn()
            flat[k] = orig - eps; lm, _ = loss_fn()
            flat[k] = orig
            num = (lp - lm) / (2*eps); ana = gflat[k]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            max_rel = max(max_rel, rel); checked += 1

    print(f"Đã kiểm tra {checked} phần tử gradient")
    print(f"Sai số tương đối lớn nhất: {max_rel:.2e}")
    print("✅ BACKPROP ĐÚNG" if max_rel < 1e-4 else "❌ Sai, kiểm tra lại backward")

if __name__ == "__main__":
    run()
