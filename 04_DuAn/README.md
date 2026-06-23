# DỰ ÁN: VISION TRANSFORMER (ViT) TỪ ĐẦU BẰNG NUMPY

Cài đặt một **Vision Transformer (encoder-only)** **chỉ bằng NumPy** — **không**
PyTorch/TensorFlow/Keras. Tự viết cả **forward** lẫn **backward** và **Adam**.

Bài toán: **phân loại ảnh tổng hợp 4 lớp** (sọc dọc / sọc ngang / ô vuông / caro).
Ảnh được **sinh tại chỗ** → không cần tải dataset, không phụ thuộc dữ liệu ngoài.

## Khác gì với bản GPT (folder `transformer-course`)?
| | GPT (decoder) | ViT (encoder) |
|--|--------------|---------------|
| Đầu vào | chuỗi token chữ | ảnh → patch → token |
| Attention | nhân-quả (có mask) | **hai chiều (không mask)** |
| Token đặc biệt | — | **[CLS]** để gom thông tin |
| Đầu ra | next-token (mỗi vị trí) | **1 nhãn lớp / ảnh** |
| Dùng lại | — | Linear, LayerNorm, Attention*, FFN, Adam |

\* Attention giống hệt, chỉ bỏ causal mask.

## Cấu trúc
```
04_DuAn/
├── README.md
├── src/
│   ├── vit.py           # LÕI: PatchEmbedding, [CLS], SelfAttention, Block, ViT, Adam
│   ├── data.py          # sinh ảnh tổng hợp 4 lớp
│   ├── gradcheck.py     # kiểm chứng backprop bằng gradient số
│   ├── train.py         # huấn luyện phân loại
│   ├── make_figures.py  # xuất biểu đồ + ảnh mẫu + attention [CLS]
│   └── quantize.py      # (mở rộng) lượng tử hoá Q8.8 — liên hệ FPGA
├── data/                # (trống — dữ liệu sinh ngẫu nhiên)
└── figures/             # hình xuất ra
```

## Yêu cầu
```bash
pip install numpy matplotlib   # matplotlib chỉ để vẽ
```

## Chạy
```bash
cd src
PYTHONUTF8=1 python gradcheck.py     # 1) kiểm chứng backprop -> sai số ~1e-6
PYTHONUTF8=1 python train.py         # 2) huấn luyện phân loại ảnh
PYTHONUTF8=1 python make_figures.py  # 3) xuất hình cho slide
PYTHONUTF8=1 python quantize.py      # 4) (mở rộng) Q8.8 / FPGA
```
> **Windows:** dùng `PYTHONUTF8=1` để console không lỗi tiếng Việt (script cũng tự ép UTF-8).
>
> **Mẹo tăng tốc:** với ma trận nhỏ, BLAS đa luồng thường *chậm hơn* do quá tải
> luồng. Đặt 1 luồng để nhanh hơn nhiều:
> ```bash
> # Git Bash / Linux:
> export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
> # PowerShell:
> $env:OPENBLAS_NUM_THREADS=1; $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1
> ```

## Thành phần đã cài (`vit.py`)
| Lớp | Forward | Backward |
|-----|:------:|:--------:|
| `PatchEmbedding` (patchify + Linear) | ✅ | ✅ |
| `[CLS]` token + Positional Embedding | ✅ | ✅ |
| `SelfAttention` (multi-head, **hai chiều**) | ✅ | ✅ |
| `FeedForward` (GELU) | ✅ | ✅ |
| `Block` (pre-norm + residual) | ✅ | ✅ |
| `ViT` (patch→encoder→[CLS]→head) | ✅ | ✅ |
| `cross_entropy` (phân loại) | ✅ | ✅ |
| `Adam` | ✅ | — |

## Siêu tham số (`train.py`)
| Tên | Mặc định | Ý nghĩa |
|-----|---------|---------|
| `IMG` | 16 | kích thước ảnh |
| `PATCH` | 4 | cạnh patch → 16 patch |
| `num_classes` | 4 | số lớp |
| `n_layer` | 3 | số khối encoder |
| `n_head` | 4 | số đầu attention |
| `n_embd` | 64 | chiều embedding D |
| `LR` | 3e-3 | tốc độ học |
| `STEPS` | 600 | số bước (hội tụ ~bước 150) |

## Hướng mở rộng
- Lớp ảnh mới (đường chéo, hình tròn); mean-pool thay [CLS]; đổi patch size.
- Ảnh nhiều kênh (RGB) — đổi `in_ch=3`.
- Positional encoding 2D; data augmentation.
- **Lượng tử hoá Q8.8 → FPGA** (liên hệ SRCNN/rPPG).

## Tham khảo
- Dosovitskiy et al., *An Image is Worth 16×16 Words* (ViT), ICLR 2021.
- Folder `../transformer-course` — nền tảng Transformer dùng lại ở đây.
