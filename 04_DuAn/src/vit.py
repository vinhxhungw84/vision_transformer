"""
vit.py
======
Vision Transformer (ViT) viết HOÀN TOÀN từ đầu, CHỈ dùng NumPy.
KHÔNG dùng PyTorch / TensorFlow / Keras / bất kỳ thư viện AI nào.

Khác với GPT (decoder-only, có mặt nạ nhân-quả), ViT là ENCODER-ONLY:
- Ảnh được cắt thành các "miếng" (patch) -> mỗi patch thành 1 token.
- Attention HAI CHIỀU (bidirectional, KHÔNG mask) -> mỗi patch nhìn mọi patch.
- Thêm 1 token đặc biệt [CLS] để gom thông tin -> phân loại.

Các khối: PatchEmbedding, [CLS] token, Positional Emb, SelfAttention (multi-head),
FeedForward, LayerNorm, Block, ViT, cross-entropy phân loại, Adam.
Tự cài cả forward và backward.

Tác giả: <điền tên bạn>
"""
import numpy as np

# ---------------------------------------------------------------------------
# 0. Cấu hình
# ---------------------------------------------------------------------------
class Config:
    def __init__(self, img_size, patch_size, num_classes, in_ch=1,
                 n_layer=3, n_head=4, n_embd=64, seed=1337):
        assert img_size % patch_size == 0, "img_size phải chia hết cho patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.n_patch = (img_size // patch_size) ** 2          # số patch
        self.patch_dim = patch_size * patch_size * in_ch       # số chiều 1 patch
        self.seq_len = self.n_patch + 1                        # +1 cho [CLS]
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        self.rng = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 1. Hàm cơ bản (giống bản GPT)
# ---------------------------------------------------------------------------
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def gelu_backward(x, dout):
    c = np.sqrt(2/np.pi)
    t = np.tanh(c * (x + 0.044715 * x**3))
    dt = (1 - t**2) * c * (1 + 3*0.044715 * x**2)
    return dout * (0.5 * (1 + t) + 0.5 * x * dt)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 2. Linear
# ---------------------------------------------------------------------------
class Linear:
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0, 0.02, size=(n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, x):
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        x = self.cache
        x2 = x.reshape(-1, x.shape[-1]); d2 = dout.reshape(-1, dout.shape[-1])
        self.dW = x2.T @ d2
        self.db = d2.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [(self, 'W', 'dW'), (self, 'b', 'db')]


# ---------------------------------------------------------------------------
# 3. LayerNorm
# ---------------------------------------------------------------------------
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim); self.beta = np.zeros(dim); self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        std = np.sqrt(var + self.eps); xhat = (x - mu) / std
        self.cache = (xhat, std)
        return self.gamma * xhat + self.beta

    def backward(self, dout):
        xhat, std = self.cache; D = xhat.shape[-1]
        self.dgamma = (dout * xhat).reshape(-1, D).sum(0)
        self.dbeta = dout.reshape(-1, D).sum(0)
        dxhat = dout * self.gamma
        dx = (1.0/std/D) * (D*dxhat - dxhat.sum(-1, keepdims=True)
                            - xhat * (dxhat*xhat).sum(-1, keepdims=True))
        return dx

    def params(self):
        return [(self, 'gamma', 'dgamma'), (self, 'beta', 'dbeta')]


# ---------------------------------------------------------------------------
# 4. Multi-Head Self-Attention (HAI CHIỀU - KHÔNG mask)
# ---------------------------------------------------------------------------
class SelfAttention:
    def __init__(self, cfg):
        self.qkv = Linear(cfg.n_embd, 3*cfg.n_embd, cfg.rng)
        self.proj = Linear(cfg.n_embd, cfg.n_embd, cfg.rng)
        self.nh = cfg.n_head; self.hs = cfg.head_size

    def forward(self, x):
        B, T, C = x.shape; nh, hs = self.nh, self.hs
        qkv = self.qkv.forward(x)
        q, k, v = np.split(qkv, 3, axis=-1)
        sh = lambda t: t.reshape(B, T, nh, hs).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hs)   # (B,nh,T,T)
        att = softmax(att, axis=-1)                          # KHÔNG mask
        y = att @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.proj.forward(y)
        self.cache = (q, k, v, att, B, T, C)
        return out

    def backward(self, dout):
        q, k, v, att, B, T, C = self.cache; nh, hs = self.nh, self.hs
        dy = self.proj.backward(dout).reshape(B, T, nh, hs).transpose(0, 2, 1, 3)
        datt = dy @ v.transpose(0, 1, 3, 2)
        dv = att.transpose(0, 1, 3, 2) @ dy
        dscore = att * (datt - (datt*att).sum(-1, keepdims=True))
        dscore = dscore / np.sqrt(hs)
        dq = dscore @ k
        dk = dscore.transpose(0, 1, 3, 2) @ q
        mg = lambda t: t.transpose(0, 2, 1, 3).reshape(B, T, C)
        dqkv = np.concatenate([mg(dq), mg(dk), mg(dv)], axis=-1)
        return self.qkv.backward(dqkv)

    def params(self):
        return self.qkv.params() + self.proj.params()


# ---------------------------------------------------------------------------
# 5. FeedForward
# ---------------------------------------------------------------------------
class FeedForward:
    def __init__(self, cfg):
        self.fc = Linear(cfg.n_embd, 4*cfg.n_embd, cfg.rng)
        self.proj = Linear(4*cfg.n_embd, cfg.n_embd, cfg.rng)

    def forward(self, x):
        self.h = self.fc.forward(x)
        return self.proj.forward(gelu(self.h))

    def backward(self, dout):
        da = self.proj.backward(dout)
        return self.fc.backward(gelu_backward(self.h, da))

    def params(self):
        return self.fc.params() + self.proj.params()


# ---------------------------------------------------------------------------
# 6. Block (pre-norm + residual)
# ---------------------------------------------------------------------------
class Block:
    def __init__(self, cfg):
        self.ln1 = LayerNorm(cfg.n_embd); self.attn = SelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd); self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x

    def backward(self, dout):
        dffn = self.ln2.backward(self.ffn.backward(dout)); dout = dout + dffn
        dattn = self.ln1.backward(self.attn.backward(dout)); dout = dout + dattn
        return dout

    def params(self):
        return (self.ln1.params() + self.attn.params()
                + self.ln2.params() + self.ffn.params())


# ---------------------------------------------------------------------------
# 7. Patch Embedding: ảnh -> chuỗi vector patch
# ---------------------------------------------------------------------------
def patchify(img, P):
    """img: (B, C, H, W) -> patches: (B, n_patch, C*P*P).
    Cắt ảnh thành lưới các ô P×P, trải phẳng mỗi ô."""
    B, C, H, W = img.shape
    nh, nw = H // P, W // P
    x = img.reshape(B, C, nh, P, nw, P)          # tách H,W thành (nh,P),(nw,P)
    x = x.transpose(0, 2, 4, 1, 3, 5)            # (B, nh, nw, C, P, P)
    x = x.reshape(B, nh*nw, C*P*P)               # (B, n_patch, patch_dim)
    return x

class PatchEmbedding:
    def __init__(self, cfg):
        self.P = cfg.patch_size
        self.proj = Linear(cfg.patch_dim, cfg.n_embd, cfg.rng)

    def forward(self, img):
        patches = patchify(img, self.P)          # (B, n_patch, patch_dim)
        return self.proj.forward(patches)        # (B, n_patch, C)

    def backward(self, dout):
        # Không cần gradient theo pixel (ảnh là dữ liệu) -> dừng ở Linear.
        # Lưu lại gradient tới từng patch để vẽ saliency (mức độ quan trọng).
        self.dpatches = dout            # (B, n_patch, D)
        self.proj.backward(dout)
        return None

    def params(self):
        return self.proj.params()


# ---------------------------------------------------------------------------
# 8. Vision Transformer
# ---------------------------------------------------------------------------
class ViT:
    def __init__(self, cfg):
        self.cfg = cfg; r = cfg.rng
        self.patch_embed = PatchEmbedding(cfg)
        self.cls_token = r.normal(0, 0.02, size=(1, 1, cfg.n_embd))     # [CLS]
        self.pos = r.normal(0, 0.02, size=(1, cfg.seq_len, cfg.n_embd)) # vị trí
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd)
        self.head = Linear(cfg.n_embd, cfg.num_classes, cfg.rng)

    def forward(self, img):
        B = img.shape[0]
        x = self.patch_embed.forward(img)                  # (B, n_patch, C)
        cls = np.broadcast_to(self.cls_token, (B, 1, self.cfg.n_embd))
        x = np.concatenate([cls, x], axis=1)               # (B, seq_len, C)
        x = x + self.pos
        for blk in self.blocks:
            x = blk.forward(x)
        x = self.ln_f.forward(x)
        cls_out = x[:, 0]                                   # token [CLS]: (B, C)
        logits = self.head.forward(cls_out)                # (B, num_classes)
        self.cache = (B,)
        return logits

    def backward(self, dlogits):
        (B,) = self.cache
        dcls_out = self.head.backward(dlogits)             # (B, C)
        # logits chỉ phụ thuộc token [CLS] -> gradient các token khác = 0
        dx = np.zeros((B, self.cfg.seq_len, self.cfg.n_embd))
        dx[:, 0] = dcls_out
        dx = self.ln_f.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        # gradient cho positional emb và [CLS]
        self.dpos = dx.sum(axis=0, keepdims=True)
        self.dcls_token = dx[:, 0:1].sum(axis=0, keepdims=True)
        # gradient cho patch embedding
        self.patch_embed.backward(dx[:, 1:])
        return None

    def params(self):
        p = [(self, 'cls_token', 'dcls_token'), (self, 'pos', 'dpos')]
        p += self.patch_embed.params()
        for blk in self.blocks:
            p += blk.params()
        p += self.ln_f.params() + self.head.params()
        return p


# ---------------------------------------------------------------------------
# 9. Cross-Entropy phân loại (logits: (B,K), targets: (B,))
# ---------------------------------------------------------------------------
def cross_entropy(logits, targets):
    B, K = logits.shape
    p = softmax(logits, axis=-1)
    loss = -np.log(p[np.arange(B), targets] + 1e-12).mean()
    dlogits = p.copy()
    dlogits[np.arange(B), targets] -= 1
    dlogits /= B
    return loss, dlogits


# ---------------------------------------------------------------------------
# 10. Adam
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        self.params = params; self.lr = lr; self.eps = eps; self.wd = weight_decay
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(getattr(o, pn)) for o, pn, _ in params]
        self.v = [np.zeros_like(getattr(o, pn)) for o, pn, _ in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (owner, pname, gname) in enumerate(self.params):
            p = getattr(owner, pname); g = getattr(owner, gname)
            if self.wd: g = g + self.wd * p
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(g*g)
            mhat = self.m[i] / (1 - self.b1**self.t)
            vhat = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
