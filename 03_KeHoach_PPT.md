# KẾ HOẠCH & DÀN Ý SLIDE: VISION TRANSFORMER TỪ ĐẦU

> ✅ **File slide đã được dựng sẵn:** `Vision_Transformer_tu_dau.pptx` (31 slide, đã
> nhúng hình thật). Tạo lại bằng: `python make_pptx.py` (cần `pip install python-pptx`).
> Tài liệu dưới đây là dàn ý gốc để bạn chỉnh/bổ sung nội dung trong file .pptx.

---


Cấu trúc bộ slide ~28 slide cho seminar 45–60 phút. Giả định người nghe đã biết
Transformer cơ bản (nếu không, thêm 3–4 slide ôn ở phần 2).

## A. Thông tin chung
- **Số slide:** ~28.
- **Công cụ:** PowerPoint / Google Slides (dễ chèn ảnh & heatmap), hoặc Marp/Beamer.
- **Phong cách:** nhiều **hình ảnh thật** (ảnh đầu vào, patch, attention map) — đây
  là lợi thế của chủ đề thị giác, hãy tận dụng.
- **Màu nhấn:** xanh dương (khối chính), cam (đường gradient/backward).
- **Ký hiệu:** thống nhất với lý thuyết (B,C,H,W,P,N,D,L,h,K).

---

## B. Dàn ý từng slide

### Phần 1 — Mở đầu (1–3)
1. **Bìa:** "Vision Transformer từ đầu bằng NumPy". Nền: ảnh chia lưới patch.
2. **Mục lục.**
3. **Câu hỏi mở:** *"Một ảnh đáng giá 16×16 chữ?"* — đặt vấn đề dùng Transformer cho ảnh.

### Phần 2 — Bối cảnh (4–7)
4. **Ôn nhanh Transformer:** attention + FFN + LN (1 slide sơ đồ).
5. **CNN vs ViT:** bảng so sánh (inductive bias, receptive field, dữ liệu).
6. **Trực giác cốt lõi:** ảnh → patch → token → Transformer (hình 3 bước).
7. **Ứng dụng ViT:** phân loại, phát hiện, phân đoạn, đa phương thức (CLIP), y tế.

### Phần 3 — Kiến trúc ViT (8–16) — phần lõi
8. **Sơ đồ tổng thể ViT** (hình lớn, tái dùng từ lý thuyết).
9. **Patchify:** minh hoạ ảnh 16×16 → 16 patch 4×4 (hình lưới + mũi tên).
10. **Patch → vector:** trải phẳng + Linear (P·P·C → D). Liên hệ "conv stride=P".
11. **Token [CLS]:** vì sao thêm; vai trò "gom thông tin" (hình token đứng đầu chuỗi).
12. **Positional Embedding:** vì sao cần vị trí; bảng (N+1)×D.
13. **Encoder block:** pre-norm `x+MHSA(LN x); x+FFN(LN x)` (sơ đồ 1 khối).
14. **Điểm KHÁC GPT:** attention **hai chiều**, KHÔNG causal mask (hình so sánh
    ma trận: tam giác vs đầy đủ).
15. **Đầu phân loại:** lấy [CLS] → LayerNorm → Linear → K lớp.
16. **Tổng kết luồng dữ liệu** + shape ở từng bước.

### Phần 4 — Huấn luyện & Backprop (17–20)
17. **Cross-entropy phân loại:** `dlogits = (p − onehot)/B`.
18. **Backprop ViT — 4 điểm khác GPT:** bắt đầu [CLS]; grad pos/[CLS]; dừng ở
    patch Linear; không mask.
19. **Gradient check:** ý tưởng + **ảnh chụp kết quả** (sai số ~1e-6). *Bằng chứng.*
20. **Adam** (ôn ngắn).

### Phần 5 — Cài đặt & Kết quả (21–27)
21. **Cấu trúc code:** sơ đồ class (PatchEmbedding, SelfAttention, Block, ViT).
    Nhấn **chỉ NumPy**.
22. **Dữ liệu tổng hợp:** 4 lớp (sọc dọc/ngang, ô vuông, caro) — ảnh ví dụ.
23. **Đường cong huấn luyện:** loss & độ chính xác (`figures/training_curve.png`).
24. **Ví dụ dự đoán:** lưới ảnh + nhãn thật/đoán (ảnh chụp).
25. **Trực quan "mô hình nhìn đâu":** bản đồ **saliency** (`figures/saliency.png`)
    — độ lớn gradient theo từng patch cho thấy vùng quan trọng (vd ô vuông sáng).
    *Ghi chú trung thực:* trên bài DỄ này attention [CLS] gần như **đồng đều**
    (phương sai ~0) vì mô hình giải qua đường residual/FFN — đây là một bài học
    hay: attention không phải lúc nào cũng "gắt"/diễn giải được; saliency hữu ích hơn.
26. **Lượng tử hoá Q8.8 / FPGA:** ViT = conv(patch) + GEMM + softmax + LN; bảng
    float vs Q8.8.
27. **Hạn chế & mở rộng:** đói dữ liệu, O(N²), Swin, DeiT, pretrain.

### Phần 6 — Kết (28)
28. **Tổng kết + Q&A:** 3 điều rút ra + link repo + tài liệu.

---

## C. Hình cần chuẩn bị (checklist)
- [ ] Sơ đồ kiến trúc ViT tổng thể.
- [ ] Minh hoạ patchify (ảnh → lưới patch).
- [ ] So sánh ma trận attention causal (tam giác) vs hai chiều (đầy đủ).
- [ ] Lưới ảnh 4 lớp tổng hợp.
- [ ] Ảnh chụp kết quả `gradcheck.py`.
- [ ] `figures/training_curve.png` (loss/accuracy).
- [ ] `figures/samples.png` (ảnh + dự đoán).
- [ ] `figures/saliency.png` (vùng quan trọng theo gradient — "mô hình nhìn đâu").

> **Xuất hình từ dự án** (matplotlib — thư viện vẽ, không phải AI):
> - `make_figures.py` đã sinh `training_curve.png`, `samples.png`, `saliency.png`.
> - Saliency: backprop từ lớp dự đoán, lấy `model.patch_embed.dpatches`, tính
>   chuẩn L2 theo chiều đặc trưng cho mỗi patch, reshape lưới (H/P)×(W/P), chồng ảnh.
> - (Tuỳ chọn) attention map: `model.blocks[L].attn.cache[3][b,head,0,1:]` — nhưng
>   trên bài dễ này gần như đồng đều (xem ghi chú slide 25).

---

## D. Lịch dựng slide (khớp timeline 2 tuần)
| Ngày | Việc |
|------|------|
| Cuối tuần 1 | Khung 28 slide + phần 1–2. |
| Đầu tuần 2 | Phần 3 (kiến trúc) từ lý thuyết. |
| Giữa tuần 2 | Phần 4–5; chèn hình thật sau khi train. |
| Cuối tuần 2 | Duyệt, luyện nói (~1.5 phút/slide). |

## E. Mẹo trình bày
- Tận dụng **ảnh thật + attention map** — chủ đề thị giác rất "ăn hình".
- Slide 14 (hai chiều vs causal) và 25 (attention [CLS]) là 2 điểm nhấn — dành thời gian.
- Nhấn "tự cài, gradient check đạt, không gọi thư viện AI" ở slide 19.
