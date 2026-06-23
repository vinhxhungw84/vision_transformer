# GIÁO ÁN: VISION TRANSFORMER TỪ ĐẦU (KHÔNG DÙNG THƯ VIỆN AI)

**Hình thức:** 5 buổi (mỗi buổi ~120 phút) + dự án xuyên suốt.
**Đối tượng:** người đã học khóa "Transformer từ đầu" (hoặc nắm attention/FFN/
LayerNorm/backprop). Biết Python + NumPy, đại số tuyến tính, đạo hàm.
**Tiên quyết quan trọng:** hiểu Multi-Head Attention & backprop ở mức tự cài được.
**Công cụ:** Python 3, NumPy (matplotlib để vẽ). **Cấm** PyTorch/TF/Keras/HF.

> Vì ViT **dùng lại** các khối Transformer, giáo án này tập trung vào phần MỚI:
> patchify, [CLS] token, encoder hai chiều, phân loại ảnh — và ít lặp lại phần
> attention đã học.

---

## Mục tiêu học tập
Kết thúc khóa, người học có thể:
1. **Giải thích** cách ViT biến ảnh thành chuỗi token (patch + [CLS] + pos).
2. **Phân biệt** ViT (encoder, hai chiều) với GPT (decoder, nhân-quả) và với CNN.
3. **Cài đặt** ViT chỉ bằng NumPy (forward + backward), huấn luyện phân loại ảnh.
4. **Kiểm chứng** backprop bằng gradient số.
5. **Trực quan hoá** attention của [CLS] và phân tích; liên hệ lượng tử hoá/FPGA.

---

## Bảng tổng quan 5 buổi
| Buổi | Chủ đề | Sản phẩm |
|------|--------|----------|
| 1 | Ôn Transformer + ý tưởng ViT + dữ liệu ảnh | Bộ sinh ảnh + hiểu kiến trúc |
| 2 | Patch Embedding + [CLS] + Positional | Forward tới chuỗi token |
| 3 | Encoder hai chiều + head + forward đủ | Forward ra logits |
| 4 | Backprop ViT + gradient check | `gradcheck.py` đạt <1e-4 |
| 5 | Train, đánh giá, trực quan attention, FPGA | Mô hình hội tụ + slide |

---

## BUỔI 1 — Ý tưởng ViT & Dữ liệu ảnh (120')
**Mục tiêu:** hiểu vì sao/ cách áp Transformer cho ảnh; chuẩn bị dữ liệu.

| Thời lượng | Hoạt động |
|-----------|-----------|
| 0–20' | Ôn nhanh Transformer (attention, FFN, LN, residual). |
| 20–45' | Giảng: ý tưởng "ảnh = chuỗi patch"; ViT vs CNN (mục 1–2). |
| 45–70' | Giảng: tổng quan kiến trúc ViT (mục 3) — vẽ sơ đồ. |
| 70–110' | **Thực hành:** viết `data.py` sinh 4 lớp ảnh tổng hợp; hiển thị vài ảnh. |
| 110–120' | Giao đọc mục 4–5. |

- **Kiểm tra:** vì sao ViT cần nhiều dữ liệu hơn CNN?
- **Bài tập:** thêm/đổi một lớp ảnh mới (vd đường chéo).

---

## BUỔI 2 — Patch Embedding, [CLS], Positional (120')
**Mục tiêu:** biến ảnh thành chuỗi token sẵn sàng cho encoder.

| Thời lượng | Hoạt động |
|-----------|-----------|
| 0–10' | Ôn dữ liệu ảnh. |
| 10–40' | Giảng: patchify (reshape/transpose), chiếu Linear (mục 4). |
| 40–65' | Giảng: token [CLS] & positional embedding (mục 5). |
| 65–110' | **Thực hành:** cài `patchify`, `PatchEmbedding`, ghép [CLS]+pos; kiểm tra shape `(B, N+1, D)`. |
| 110–120' | Tổng kết. |

- **Kiểm tra:** ảnh 32×32, P=8 → bao nhiêu patch? chuỗi dài bao nhiêu?
- **Bài tập:** xác nhận patchify đảo ngược được (un-patchify) để hiểu cấu trúc.

---

## BUỔI 3 — Encoder hai chiều + Head (120')
**Mục tiêu:** ghép encoder (không mask) và đầu phân loại; forward đủ.

| Thời lượng | Hoạt động |
|-----------|-----------|
| 0–10' | Ôn buổi 2. |
| 10–35' | Giảng: attention hai chiều (khác causal); khối encoder pre-norm (mục 6). |
| 35–55' | Giảng: lấy [CLS] → head phân loại (mục 7). |
| 55–105' | **Thực hành:** cài `SelfAttention` (không mask), `Block`, `ViT.forward`; kiểm tra logits `(B,K)`. |
| 105–120' | Chạy forward end-to-end với ảnh thật từ `data.py`. |

- **Kiểm tra:** điểm khác duy nhất giữa attention của ViT và GPT là gì?
- **Bài tập:** thử mean-pooling thay cho [CLS], so sánh.

---

## BUỔI 4 — Backprop ViT & Gradient Check (120')  ⭐
**Mục tiêu:** cài backward; kiểm chứng.

| Thời lượng | Hoạt động |
|-----------|-----------|
| 0–25' | Giảng: 4 điểm khác bản GPT (mục 9): bắt đầu từ [CLS], grad pos/[CLS], dừng ở patch Linear, không mask. |
| 25–50' | Ôn nhanh backward Linear/LN/Attention/FFN (tái dùng). |
| 50–100' | **Thực hành:** cài `ViT.backward`; viết & chạy `gradcheck.py`. |
| 100–120' | Debug đến sai số <1e-4. |

- **Mẹo:** kiểm tra riêng patch embedding & head trước, rồi mới toàn mô hình.
- **Bài tập:** giải thích vì sao gradient các token ≠ [CLS] (qua head) bằng 0.

---

## BUỔI 5 — Train, Đánh giá, Trực quan, FPGA (120')
**Mục tiêu:** huấn luyện hội tụ; diễn giải; liên hệ phần cứng; trình bày.

| Thời lượng | Hoạt động |
|-----------|-----------|
| 0–15' | Giảng: cross-entropy phân loại; vòng huấn luyện; Adam (ôn). |
| 15–55' | **Thực hành:** cài `train.py`; huấn luyện 4 lớp ảnh đến hội tụ. |
| 55–80' | **Thực hành:** trực quan attention [CLS] (heatmap); nhận xét patch nào quan trọng. |
| 80–100' | Lượng tử hoá Q8.8, đo sụt chính xác (liên hệ SRCNN/rPPG FPGA). |
| 100–120' | Trình bày kết quả (slide). |

- **Sản phẩm cuối:** repo chạy được + slide + báo cáo.

---

## Đánh giá (gợi ý)
| Hạng mục | Tỷ lệ |
|----------|------|
| Patch embedding + [CLS] + pos đúng shape | 15% |
| Forward ViT đầy đủ | 15% |
| Backprop đúng (gradient check đạt) | 30% |
| Train hội tụ + đánh giá | 25% |
| Trực quan attention + báo cáo/slide | 15% |

## Lỗi thường gặp & cách xử lý
- **Sai shape khi patchify** → in shape từng bước; nhớ thứ tự transpose.
- **Quên thêm [CLS] vào pos** → pos phải dài N+1, không phải N.
- **Gradient [CLS]/pos sai** → nhớ cộng theo batch (`sum(axis=0)`).
- **Loss không giảm** → lr; quên weight decay; ảnh chưa chuẩn hoá.
- **Console Windows lỗi tiếng Việt** → `PYTHONUTF8=1`.
- **Nhầm tưởng cần mask** → ViT KHÔNG dùng causal mask.
