# Checklist Lab Day 10

## Tiến độ theo GUIDE

- [x] Bước 0: môi trường Colab T4, `pytest` của đề pass (68 passed, 1 skipped), test riêng `day10/tests` pass
- [x] Bước 1: dữ liệu `d10-21e4070301`, split 1,422 / 276 / 311 ảnh, không có phiên nào nằm ở hai split
- [x] Bước 2: YOLO → JSON thang 0–1000, sắp box theo (y1, x1), quy ngược về pixel lệch trung vị 0.24 px
- [x] Bước 3: E0 zero-shot → `report/evidence/e0.json` (F1 0.088, parse_valid_rate 0.842)
- [x] Bước 4: mask loss: 220 token ảnh, chỉ phần JSON + `<|im_end|>` được tính loss
- [x] Bước 5: LoRA 252 module (E1, E2) và 348 module (E3); E2 − E1 = 109,105,152 tham số projector
- [x] Bước 6: huấn luyện 300 bước, lưu adapter và projector (`extra_trainable.pt`), có `run.json`
- [x] Bước 7: E1, E2, E3 → `e1.json`, `e2.json`, `e3.json`, có `meta` (tham số, VRAM, phút)
- [x] Bước 8: E4 `data_size` 0.1 và 1.0 → `e4_data_0.1.json`, `e4_data_1.0.json`
- [x] Bước 9 (bonus): reward IoU có test, GRPO 60 bước → `e5.json`, `report/e5_grpo_log.json`

## Trước khi nộp

- [x] `REPORT.md` có họ tên, `dataset_version`, bảng tổng hợp, 10 câu và phần nhìn lại, không còn `<điền`
- [x] ≥ 6 ảnh dự đoán đúng/sai có box GT (xanh) và dự đoán (đỏ): `report/figures/e3_good_*.jpg`, `e3_bad_*.jpg`
- [x] Số trong bảng khớp `report/evidence/*.json` (chạy lại `scripts/score.py` trên máy ra cùng kết quả)
- [x] Không commit `data/`, `work/`, `.env`, `kaggle.json`, trọng số (`*.safetensors`, `*.bin`, `*.pt`)
- [x] Không sửa `scripts/` và `tests/` của đề
- [x] ≥ 5 commit theo bước (`git rev-list --count HEAD`)
- [x] `python scripts/check_submission.py`: 0 FAIL, 0 WARN
- [x] Repo GitHub ở chế độ Public
