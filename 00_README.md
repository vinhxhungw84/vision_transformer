# KHÓA HỌC: VISION TRANSFORMER (ViT) TỪ ĐẦU — KHÔNG DÙNG THƯ VIỆN AI

Trọn bộ tài liệu học & dạy về **Vision Transformer**, kèm dự án cài đặt **hoàn
toàn bằng NumPy**. Là phần nối tiếp của khóa `transformer-course` (Transformer cho
NLP) — ở đây áp Transformer cho **ảnh**.

> **Nên học `transformer-course` trước.** ViT dùng lại y nguyên Attention/FFN/
> LayerNorm/Adam; phần MỚI chỉ là: cắt patch, token [CLS], encoder hai chiều,
> phân loại ảnh.

## Nội dung
| File/Thư mục | Mô tả |
|--------------|-------|
| `01_LyThuyet.md` | **Lý thuyết** ViT: patch embedding, [CLS], pos, encoder hai chiều, ViT vs CNN, backprop khác GPT, FPGA. |
| `02_GiaoAn.md` | **Giáo án 5 buổi** (mục tiêu, hoạt động theo phút, đánh giá, bài tập). |
| `03_KeHoach_PPT.md` | **Dàn ý ~28 slide** + checklist hình + lịch dựng. |
| `Vision_Transformer_tu_dau.pptx` | **File slide hoàn chỉnh (31 slide)**, đã nhúng hình. Tạo lại: `python make_pptx.py`. |
| `04_DuAn/` | **Dự án code** ViT NumPy: forward+backward, train, gradient check, lượng tử hoá, vẽ hình. |
| `05_Timeline_2Tuan.md` | **Timeline 14 ngày** (rút còn ~10 ngày nếu đã làm khóa Transformer). |

## Bắt đầu nhanh (dự án)
```bash
cd 04_DuAn/src
PYTHONUTF8=1 python gradcheck.py     # 1) kiểm chứng backprop  -> sai số ~1e-6
PYTHONUTF8=1 python train.py         # 2) huấn luyện phân loại ảnh
PYTHONUTF8=1 python make_figures.py  # 3) xuất hình cho slide
PYTHONUTF8=1 python quantize.py      # 4) (mở rộng) Q8.8 / FPGA
```

## Bài toán minh hoạ
Phân loại ảnh xám 16×16 thành 4 lớp: **sọc dọc / sọc ngang / ô vuông / caro**
(sinh tại chỗ, không cần dataset). Patch 4×4 → 16 patch + 1 token [CLS].

## Lộ trình đề xuất
1. Đọc `05_Timeline_2Tuan.md`.
2. Học theo `02_GiaoAn.md`, tra cứu `01_LyThuyet.md`.
3. Cài/chạy `04_DuAn/` song song.
4. Dựng slide theo `03_KeHoach_PPT.md`, chèn hình thật từ `figures/`
   (đường cong loss, ảnh mẫu, **bản đồ saliency** "mô hình nhìn đâu").

> Triết lý: **mọi phép toán hiển lộ trong mã** — hiểu ViT từ pixel đến logits.
