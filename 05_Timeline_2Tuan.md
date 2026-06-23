# TIMELINE 2 TUẦN — HOÀN THIỆN VISION TRANSFORMER TỪ ĐẦU

Kế hoạch 14 ngày cho cả 4 sản phẩm: lý thuyết, giáo án, slide, dự án code.
~2–3 giờ/ngày. **Lưu ý:** nếu đã làm khóa "Transformer từ đầu", phần lõi
(attention/FFN/LN/backprop) **tái dùng được** → ViT nhanh hơn, có thể rút còn ~10 ngày.

> 🧠 lý thuyết · 💻 code · 📊 slide · 📝 viết · ✅ mốc nghiệm thu

---

## TỔNG QUAN
```
TUẦN 1 — HIỂU + CÀI ĐẶT
 T2  Ôn Transformer + ý tưởng ViT + dữ liệu ảnh
 T3  Patch embedding + [CLS] + positional
 T4  Encoder hai chiều + head -> forward đủ
 T5  Backprop ViT + gradient check  ⭐
 T6  Train phân loại ảnh + đánh giá
 T7  Trực quan attention + xuất hình   ✅ mốc 1: code chạy & hội tụ
 CN  Đệm + khung slide

TUẦN 2 — ĐÀO SÂU + MỞ RỘNG + TRÌNH BÀY
 T8  Hoàn thiện lý thuyết
 T9  Mở rộng: lớp ảnh mới / mean-pool / patch size khác
 T10 Lượng tử hoá Q8.8 (liên hệ FPGA)
 T11 Slide phần kiến trúc
 T12 Slide kết quả + chèn hình thật
 T13 Báo cáo + giáo án
 T14 Duyệt tổng + luyện nói          ✅ mốc 2: bàn giao đầy đủ
```

---

## CHI TIẾT TỪNG NGÀY

### TUẦN 1

**Ngày 1 (T2) — Ý tưởng & Dữ liệu**
- 🧠 Đọc lý thuyết mục 1–3 (ý tưởng, ViT vs CNN, kiến trúc).
- 💻 Viết `data.py` sinh 4 lớp ảnh; hiển thị/kiểm tra vài ảnh.
- ✅ Hiểu vì sao "ảnh = chuỗi patch".

**Ngày 2 (T3) — Patch Embedding + [CLS] + Pos**
- 🧠 Đọc mục 4–5.
- 💻 Cài `patchify`, `PatchEmbedding`, ghép [CLS] + positional.
- ✅ Forward tới chuỗi token, shape `(B, N+1, D)` đúng.

**Ngày 3 (T4) — Encoder + Head**
- 🧠 Đọc mục 6–7.
- 💻 Cài `SelfAttention` (KHÔNG mask), `Block`, `ViT.forward`, head.
- ✅ Forward end-to-end ra logits `(B, K)`.

**Ngày 4 (T5) — Backprop + Gradient check ⭐**
- 🧠 Đọc mục 9 (4 điểm khác GPT).
- 💻 Cài `ViT.backward`; viết & chạy `gradcheck.py`.
- ✅ **Sai số gradient < 1e-4** (mốc khó nhất).

**Ngày 5 (T6) — Train + Đánh giá**
- 🧠 Đọc mục 8.
- 💻 Cài `cross_entropy`, vòng `train.py`; huấn luyện đến hội tụ.
- ✅ Độ chính xác tăng rõ theo bước.

**Ngày 6 (T7) — Trực quan + Hình**
- 💻 `make_figures.py`: đường cong loss, lưới ảnh mẫu, **attention [CLS]**.
- ✅ **MỐC 1:** code hoàn chỉnh, hội tụ (mục tiêu ≥ 95% trên 4 lớp tổng hợp).

**Ngày 7 (CN) — Đệm + khung slide**
- 📊 Dựng khung 28 slide; điền phần 1–2.

---

### TUẦN 2

**Ngày 8 (T8) — Hoàn thiện lý thuyết**
- 📝 Rà `01_LyThuyet.md`, thêm ví dụ số (đếm patch, shape), kiểm công thức.

**Ngày 9 (T9) — Mở rộng**
- 💻 Thêm lớp ảnh mới (đường chéo/hình tròn); thử **mean-pool** thay [CLS];
  đổi `patch_size`, so sánh số patch & độ chính xác.

**Ngày 10 (T10) — Lượng tử hoá Q8.8**
- 💻 `quantize.py`: float → Q8.8, đo sụt chính xác.
- 🧠 So quy trình với `compare_srcnn.py`; ghi điểm nghẽn (softmax/LN, O(N²)).

**Ngày 11 (T11) — Slide kiến trúc**
- 📊 Phần 3 (patchify, [CLS], pos, encoder hai chiều, head).

**Ngày 12 (T12) — Slide kết quả**
- 📊 Phần 4–5; chèn hình thật: gradient check, loss, ảnh mẫu, attention [CLS].

**Ngày 13 (T13) — Báo cáo + giáo án**
- 📝 Báo cáo 4–6 trang; rà `02_GiaoAn.md`.

**Ngày 14 (T14) — Duyệt tổng + luyện nói**
- 📊 Luyện thuyết trình; chạy lại repo từ máy sạch.
- ✅ **MỐC 2:** bàn giao đủ 4 sản phẩm.

---

## CHECKLIST BÀN GIAO
- [ ] `01_LyThuyet.md`, `02_GiaoAn.md`, `03_KeHoach_PPT.md`, `05_Timeline_2Tuan.md`.
- [ ] `04_DuAn/` chạy: `gradcheck.py` đạt, `train.py` hội tụ, `quantize.py`, `make_figures.py`.
- [ ] `figures/`: training_curve, samples, attention_cls.
- [ ] File `.pptx` ~28 slide + báo cáo PDF + README.

## ƯU TIÊN KHI THIẾU GIỜ
1. **Phải có:** forward + backward đúng (gradient check) + train hội tụ.
2. **Nên có:** trực quan attention [CLS] (điểm nhấn của ViT) + slide.
3. **Có thì tốt:** Q8.8/FPGA, lớp ảnh mở rộng, mean-pool.

> Nếu đã có khóa Transformer: tái dùng `Linear/LayerNorm/SelfAttention/FFN/Adam`,
> chỉ viết MỚI: `patchify`, `PatchEmbedding`, ghép [CLS]+pos, `ViT.forward/backward`,
> bỏ causal mask. Đó là toàn bộ phần "mới" của ViT.
