"""
make_figures.py
===============
Sinh hình cho slide/báo cáo (matplotlib — thư viện VẼ, không phải AI):
  1) figures/training_curve.png  — loss & độ chính xác
  2) figures/samples.png         — lưới ảnh + nhãn thật/đoán
  3) figures/attention_cls.png   — [CLS] chú ý patch nào (chồng lên ảnh)

Chạy:  PYTHONUTF8=1 python make_figures.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vit import ViT, Config, cross_entropy, Adam
from data import make_batch, CLASS_NAMES
from train import IMG, PATCH, evaluate

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

def train_and_record(steps=600):
    rng = np.random.default_rng(0)
    cfg = Config(img_size=IMG, patch_size=PATCH, num_classes=4,
                 n_layer=3, n_head=4, n_embd=64, seed=1337)
    model = ViT(cfg)
    opt = Adam(model.params(), lr=3e-3, betas=(0.9, 0.95), weight_decay=1e-4)
    hist = {"step": [], "loss": [], "acc": []}
    for step in range(1, steps+1):
        X, y = make_batch(rng, 64, IMG)
        loss, dl = cross_entropy(model.forward(X), y)
        model.backward(dl); opt.step()
        if step % 50 == 0 or step == 1:
            acc = evaluate(model, np.random.default_rng(123), n=300)
            hist["step"].append(step); hist["loss"].append(loss); hist["acc"].append(acc)
    return model, cfg, hist

def plot_curve(hist):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(hist["step"], hist["loss"], color="tab:blue")
    ax1.set_xlabel("Bước"); ax1.set_ylabel("Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(hist["step"], [a*100 for a in hist["acc"]], color="tab:orange")
    ax2.set_ylabel("Độ chính xác (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    plt.title("Huấn luyện Vision Transformer (NumPy) — phân loại ảnh")
    fig.tight_layout(); p = os.path.join(OUT, "training_curve.png")
    plt.savefig(p, dpi=120); plt.close(); print("Đã lưu", p)

def plot_samples(model):
    rng = np.random.default_rng(7)
    X, y = make_batch(rng, 8, IMG)
    pred = model.forward(X).argmax(axis=1)
    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X[i, 0], cmap="gray"); ax.axis("off")
        ok = "OK" if pred[i] == y[i] else "X"
        ax.set_title(f"{CLASS_NAMES[y[i]]}\n→{CLASS_NAMES[pred[i]]} [{ok}]", fontsize=9)
    fig.suptitle("Ví dụ dự đoán ViT")
    fig.tight_layout(); p = os.path.join(OUT, "samples.png")
    plt.savefig(p, dpi=120); plt.close(); print("Đã lưu", p)

def report_attention(model):
    """In phương sai attention [CLS]->patch để minh hoạ: trên bài DỄ này,
    mô hình giải bằng đường residual/FFN nên attention gần như ĐỒNG ĐỀU."""
    vmax = 0.0
    for blk in model.blocks:
        att = blk.attn.cache[3]
        vmax = max(vmax, att[0, :, 0, 1:].var(axis=-1).max())
    print(f"[Ghi chú] Phương sai attention [CLS]->patch lớn nhất = {vmax:.2e} "
          f"(~0 => attention gần như đồng đều; xem giải thích trong README).")

def plot_saliency(model, cfg):
    """Bản đồ SALIENCY: độ lớn gradient của loss theo từng patch -> mô hình
    'quan tâm' vùng nào. Trung thực & hữu ích hơn attention (vốn đồng đều ở đây)."""
    from data import make_image
    rng = np.random.default_rng(3)
    img, cls = make_image(rng, IMG, cls=2)        # ảnh lớp "ô vuông"
    X = img[None, None]
    logits = model.forward(X)
    _, dlogits = cross_entropy(logits, np.array([cls]))
    model.backward(dlogits)                        # backprop -> lưu dpatches
    g = model.patch_embed.dpatches[0]              # (n_patch, D)
    sal = np.linalg.norm(g, axis=-1)               # độ quan trọng mỗi patch
    grid = int(np.sqrt(cfg.n_patch))
    heat = np.kron(sal.reshape(grid, grid), np.ones((PATCH, PATCH)))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Ảnh gốc (ô vuông)"); axes[0].axis("off")
    axes[1].imshow(heat, cmap="viridis"); axes[1].set_title("Saliency (|grad| mỗi patch)"); axes[1].axis("off")
    axes[2].imshow(img, cmap="gray"); axes[2].imshow(heat, cmap="jet", alpha=0.5)
    axes[2].set_title("Chồng lớp"); axes[2].axis("off")
    fig.tight_layout(); p = os.path.join(OUT, "saliency.png")
    plt.savefig(p, dpi=120); plt.close(); print("Đã lưu", p)

if __name__ == "__main__":
    model, cfg, hist = train_and_record()
    plot_curve(hist); plot_samples(model)
    report_attention(model)
    plot_saliency(model, cfg)
    print("Xong. Mở thư mục figures/ để xem.")
