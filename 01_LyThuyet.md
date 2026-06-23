# LÝ THUYẾT VISION TRANSFORMER (ViT) — TỪ ĐẦU, KHÔNG DÙNG THƯ VIỆN AI

> Tài liệu lý thuyết đi kèm dự án cài đặt ViT bằng NumPy thuần.
> **Tiên quyết:** nên đọc trước khóa "Transformer từ đầu" (folder
> `transformer-course`) — ViT dùng lại đúng các khối Attention/FFN/LayerNorm,
> chỉ khác ở cách **đưa ảnh vào** và là **encoder-only**.

---

## Mục lục
1. [Từ NLP sang thị giác: ý tưởng cốt lõi](#1-ý-tưởng)
2. [So sánh ViT với CNN](#2-vit-vs-cnn)
3. [Tổng quan kiến trúc ViT](#3-kiến-trúc)
4. [Patch Embedding: biến ảnh thành chuỗi](#4-patch-embedding)
5. [Token [CLS] & Positional Embedding](#5-cls--pos)
6. [Encoder: Attention hai chiều + FFN](#6-encoder)
7. [Đầu phân loại (Classification head)](#7-head)
8. [Hàm mất mát & huấn luyện](#8-loss)
9. [Backprop những điểm khác GPT](#9-backprop)
10. [Chi phí, dữ liệu & mẹo huấn luyện](#10-chi-phí)
11. [Liên hệ phần cứng / FPGA & xử lý ảnh](#11-fpga)
12. [Thuật ngữ Anh–Việt](#12-thuật-ngữ)

---

## 1. Ý tưởng

Transformer thống trị NLP nhờ attention. **Vision Transformer**
(Dosovitskiy et al., 2020 — *"An Image is Worth 16×16 Words"*) đặt câu hỏi:
*nếu coi mỗi MIẾNG ảnh như một "từ", có thể dùng y nguyên Transformer cho ảnh
không?* Câu trả lời: **có**, và khi đủ dữ liệu, ViT vượt CNN.

Ý tưởng 3 bước:
1. **Cắt ảnh thành các patch** vuông (vd 16×16 pixel), trải phẳng mỗi patch.
2. **Chiếu tuyến tính** mỗi patch thành một vector (token) → ảnh thành *chuỗi token*.
3. Đưa chuỗi đó qua **Transformer encoder** y như xử lý câu chữ.

> "An image is worth 16×16 words" = một ảnh đáng giá bằng các "từ" 16×16 pixel.

---

## 2. ViT vs CNN

| Tiêu chí | CNN | ViT |
|----------|-----|-----|
| Thiên kiến quy nạp (inductive bias) | mạnh: tính cục bộ, bất biến tịnh tiến | yếu: học gần như từ đầu |
| Trường nhìn (receptive field) | lớn dần theo độ sâu | **toàn cục ngay lớp đầu** (attention) |
| Cần dữ liệu | ít cũng tốt | **cần nhiều** (hoặc pretrain) mới vượt CNN |
| Chi phí theo độ phân giải | tuyến tính | **O(N²)** theo số patch |
| Khả năng diễn giải | feature map | bản đồ attention (xem patch nào quan trọng) |

Ý chính: CNN "gài sẵn" giả định về ảnh (lân cận quan trọng) nên học nhanh với
ít dữ liệu; ViT linh hoạt hơn nhưng phải **tự học** các giả định đó → đói dữ liệu.

---

## 3. Kiến trúc

```
        Ảnh (B, C, H, W)
              │  cắt patch P×P
        ┌─────▼──────┐
        │ Patchify   │  -> (B, N, P·P·C)   N = (H/P)·(W/P) patch
        │ Linear     │  -> (B, N, D)       chiếu mỗi patch thành vector D chiều
        └─────┬──────┘
              │  thêm [CLS] token ở đầu  -> (B, N+1, D)
              │  + Positional Embedding
        ┌─────▼──────┐
        │  Encoder   │  lặp L lần
        │ ┌────────┐ │
        │ │LN→MHSA │ │ + residual   (Attention HAI CHIỀU, không mask)
        │ │LN→FFN  │ │ + residual
        │ └────────┘ │
        └─────┬──────┘
        ┌─────▼──────┐
        │ LayerNorm  │
        │ lấy token  │  -> vector [CLS]: (B, D)
        │ [CLS]      │
        │ Linear → K │  -> logits (B, num_classes)
        └─────┬──────┘
              ▼   softmax -> xác suất lớp
```

Ký hiệu: **B** batch, **C** kênh ảnh (1 nếu xám), **H,W** cao/rộng, **P** cạnh
patch, **N**=(H/P)(W/P) số patch, **D**=`n_embd`, **L** số lớp, **h** số đầu, **K** số lớp phân loại.

---

## 4. Patch Embedding

### Cắt patch (patchify)
Ảnh `(C, H, W)` được chia thành lưới `(H/P)×(W/P)` ô, mỗi ô `P×P`. Mỗi ô được
**trải phẳng** thành vector dài `C·P·P`. Ví dụ ảnh 16×16 xám, P=4:
- số patch N = (16/4)·(16/4) = **16**
- mỗi patch dài 1·4·4 = **16**

Trong code, đây chỉ là **reshape + transpose** (xem `patchify`), không có tham số.

### Chiếu tuyến tính
Một lớp `Linear: P·P·C → D` biến mỗi patch thành token D chiều. (Tương đương một
phép tích chập kernel P, stride P — đó là cách cài thường thấy, nhưng ở đây ta
làm tường minh bằng reshape + Linear.)

Kết quả: ảnh → **chuỗi N token**, từ đây giống hệt xử lý câu trong NLP.

---

## 5. Token [CLS] & Positional Embedding

### Token [CLS]
Ta thêm **một token học được** `[CLS]` (classification) vào **đầu** chuỗi →
chuỗi dài N+1. Sau encoder, **chỉ lấy vector tại vị trí [CLS]** làm đại diện
toàn ảnh để phân loại. Nhờ attention hai chiều, [CLS] "đọc" được mọi patch và
gom thông tin lại. (Đây là tham số, khởi tạo ngẫu nhiên, học cùng mô hình.)

> *Biến thể:* thay vì [CLS] có thể lấy **trung bình** (mean-pooling) toàn bộ
> token đầu ra. Cả hai đều phổ biến.

### Positional Embedding
Như mọi Transformer, attention **không biết thứ tự/vị trí**. Ta cộng một bảng
vị trí học được `pos ∈ R^{(N+1)×D}` vào chuỗi. Nhờ đó mô hình biết patch nào ở
góc trên-trái, patch nào ở giữa… (quan trọng cho bài toán phụ thuộc vị trí, vd
"ô vuông sáng nằm ở đâu").

---

## 6. Encoder: Attention hai chiều + FFN

Mỗi khối encoder **giống hệt** khối Transformer đã học, **trừ một điểm**:
> **KHÔNG có causal mask.** Mỗi patch được nhìn **mọi** patch khác (kể cả "sau"
> nó), vì ảnh không có chiều thời gian — đây là attention **hai chiều**.

Khối (pre-norm):
```
x = x + MHSA( LN(x) )      # Multi-Head Self-Attention, không mask
x = x + FFN ( LN(x) )      # Linear → GELU → Linear
```

- **MHSA:** `softmax(QKᵀ/√d)·V` chia thành h đầu (xem lý thuyết GPT mục 4–5).
- **FFN:** mở rộng D→4D→D.
- **Residual + LayerNorm:** ổn định và giúp gradient (như GPT).

Bản đồ attention của [CLS] (hàng đầu của ma trận `(N+1)×(N+1)`) cho biết **[CLS]
chú ý patch nào nhất** → công cụ diễn giải trực quan.

> **Lưu ý thực nghiệm (trung thực):** trên bài toán *dễ* trong dự án này, attention
> [CLS] đo được gần như **đồng đều** (phương sai ~0) — mô hình giải quyết qua đường
> residual + FFN, gần như không cần attention "gắt". Đây là bài học quan trọng:
> **attention không phải lúc nào cũng diễn giải được**. Khi đó, **bản đồ saliency**
> (độ lớn gradient theo từng patch, xem `make_figures.py` → `figures/saliency.png`)
> phản ánh "mô hình nhìn đâu" tốt hơn. Trên bài khó/dữ liệu thật, attention thường
> chuyên biệt hoá rõ hơn.

---

## 7. Đầu phân loại (head)

Sau L khối encoder:
```
x   = LayerNorm(x)
cls = x[:, 0]              # lấy token [CLS]: (B, D)
logits = cls · W_head + b  # (B, K)
```
`softmax(logits)` cho xác suất từng lớp. Vì logits **chỉ phụ thuộc token [CLS]**,
khi lan truyền ngược, gradient vào các token patch khác (qua head) bằng 0 — chúng
chỉ nhận gradient *gián tiếp* qua attention ở các lớp dưới (xem mục 9).

---

## 8. Loss & Training

### Cross-Entropy phân loại
Với nhãn lớp `y ∈ {0,…,K-1}`:
```
p = softmax(logits)
L = -log p[y]          (trung bình trên batch)
dlogits = (p - onehot(y)) / B
```
(Khác bản GPT: nhãn cho **cả ảnh** một con số, không phải mỗi token một nhãn.)

### Vòng lặp
```
for mỗi bước:
    logits = model.forward(X)          # X: (B,1,H,W)
    loss, dlogits = cross_entropy(logits, y)
    model.backward(dlogits)
    optimizer.step()                   # Adam
```

---

## 9. Backprop — những điểm KHÁC bản GPT

Đa số công thức giống hệt (Linear, LayerNorm, Attention, FFN, Adam — xem lý
thuyết GPT mục 10). Ba điểm riêng của ViT:

1. **Bắt đầu từ token [CLS]:** chỉ `x[:,0]` có gradient từ head; ta tạo
   `dx` toàn 0 rồi gán `dx[:,0] = head.backward(dlogits)` trước khi đẩy ngược
   qua `ln_f` và các block.

2. **Gradient cho [CLS] token & pos embedding:**
   ```
   dpos       = Σ_batch dx                  (cộng theo batch)
   dcls_token = Σ_batch dx[:, 0]
   ```

3. **Patch embedding dừng tại Linear:** ảnh là *dữ liệu đầu vào*, không cần
   gradient theo pixel → backward chỉ tới lớp `Linear` của patch embedding rồi
   dừng (không cần "un-patchify").

4. **Không có mask** trong backward attention (bỏ bước zero-hoá ô bị che).

> Kiểm chứng: `src/gradcheck.py` cho sai số ~1e-6 ⇒ đúng.

---

## 10. Chi phí, dữ liệu & mẹo

- **Chi phí attention O(N²)** theo số patch. Ảnh phân giải cao → N lớn → tốn.
  Giải pháp: patch lớn hơn, hoặc kiến trúc phân cấp (Swin Transformer).
- **ViT đói dữ liệu:** trên dữ liệu nhỏ thường thua CNN; thường phải **pretrain**
  trên tập lớn rồi fine-tune. (Trong dự án này ta dùng *dữ liệu tổng hợp vô hạn*
  nên không gặp vấn đề thiếu dữ liệu.)
- **Mẹo ổn định:** pre-norm, warmup learning rate, weight decay, augmentation.
- **Chọn patch size:** nhỏ → nhiều token, chi tiết hơn nhưng tốn; lớn → ngược lại.

---

## 11. Liên hệ FPGA / xử lý ảnh

(Liên hệ hướng FPGA của bạn — rPPG/SRCNN.)
- Patch embedding = **tích chập kernel=stride=P** → ánh xạ thẳng sang phần cứng
  conv bạn đã làm; phần còn lại là **GEMM + softmax + LayerNorm**.
- Lượng tử hoá **Q8.8** toàn bộ trọng số (xem `quantize.py`) — đúng quy trình
  `compare_srcnn.py` (so float vs Q8.8). Softmax/LayerNorm cần LUT/xấp xỉ.
- ViT xử lý **ảnh** → liên hệ trực tiếp pipeline camera (OV7670/RGB565) trong dự
  án rPPG: tiền xử lý → cắt patch → encoder.
- Điểm nghẽn O(N²): với ảnh lớn cần KV-cache / attention tuyến tính / Swin.

---

## 12. Thuật ngữ Anh–Việt

| English | Tiếng Việt |
|---------|-----------|
| Patch | miếng/ô ảnh |
| Patch embedding | nhúng patch |
| [CLS] token | token phân loại |
| Class token / classification head | đầu phân loại |
| Bidirectional attention | chú ý hai chiều |
| Encoder-only | chỉ bộ mã hoá |
| Inductive bias | thiên kiến quy nạp |
| Receptive field | trường tiếp nhận |
| Mean pooling | gộp trung bình |
| Patchify / unfold | cắt thành patch |

---

## Tài liệu tham khảo
1. Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image
   Recognition at Scale*, ICLR 2021.
2. Vaswani et al., *Attention Is All You Need*, 2017.
3. Liu et al., *Swin Transformer*, 2021 (kiến trúc phân cấp cho ảnh lớn).
4. Folder `transformer-course/01_LyThuyet.md` — nền tảng Transformer.
