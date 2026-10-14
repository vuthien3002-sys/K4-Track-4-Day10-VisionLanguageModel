# Báo cáo các bẫy kỹ thuật đã gặp

Mỗi mục ghi: triệu chứng, nguyên nhân, cách xử lý trong `day10/`, và cách mình kiểm chứng.

## 1. `llm_int8_skip_modules=["visual", "lm_head"]` không giữ ViT ở fp16

- **Triệu chứng:** không có lỗi nào. ViT và projector âm thầm bị lượng tử hoá 4-bit, nên E2/E3 không huấn luyện đầy đủ projector được.
- **Nguyên nhân:** transformers 5.x (`quantizers_utils.should_convert_module`) so pattern bằng `re.match` từ đầu tên module. Tên thật là `model.visual.blocks.0.attn.qkv`, nên `"visual"` không khớp; `"lm_head"` khớp vì trùng khít tên.
- **Xử lý:** `SKIP_MODULES = ["model.visual", "lm_head"]` trong `day10/vlm.py`, cộng hàm `check_quantization()` báo lỗi nếu ngoài LLM còn có `Linear4bit`.
- **Kiểm chứng:** trên một model Qwen3-VL thu nhỏ, `"visual"` để lại 20/20 Linear của ViT bị lượng tử hoá, còn `"model.visual"` thì 0. Trên Colab, kết quả là `{'llm': 252}`.

## 2. Thiếu `mm_token_type_ids` trong batch

- **Triệu chứng:** `ValueError: Multimodal data was passed … but mm_token_type_ids is missing`.
- **Nguyên nhân:** Qwen3-VL trong transformers 5.x cần trường này để tính M-RoPE cho token ảnh. Mã `encode_example`/`collate` trên trang đề bỏ trường này.
- **Xử lý:** `encode_example` giữ lại trường này, `collate` pad nó bằng 0.

## 3. Chọn module LoRA theo hậu tố khớp nhầm projector

- **Triệu chứng:** E3 ra 332 module thay vì 348.
- **Nguyên nhân:** danh sách `["qkv", "linear_fc1", "linear_fc2"]` khớp cả `visual.merger.linear_fc1/2` và `deepstack_merger_list.N.linear_fc1/2`, đồng thời thiếu `attn.proj`.
- **Xử lý:** regex trên tên đầy đủ (PEFT dùng `re.fullmatch`): `LLM_TARGETS`, `VIT_TARGETS` trong `day10/vlm.py`. `run_experiment` báo lỗi nếu số module khác 252/348.

## 4. Projector phải ở float32

- `GradScaler.unscale_()` từ chối gradient fp16. Ngoài ra, cập nhật nhỏ (lr cỡ 2e-4, rồi giảm dần về 0 theo cosine) bị làm tròn mất ở fp16.
- **Xử lý:** `apply_setup` đưa projector về float32 và kiểm tra mọi tham số huấn luyện đều là float32. Khi suy luận phải bọc `torch.autocast(fp16)`, vì projector float32 nhận đầu vào fp16.

## 5. `save_pretrained()` của PEFT không lưu projector

- **Xử lý:** lưu riêng `extra_trainable.pt` mỗi 100 bước. `load_trained()` nạp lại, kiểm tra `unexpected_keys` rỗng và so sánh giá trị một tensor. Nếu tên lệch, `load_state_dict(strict=False)` sẽ âm thầm bỏ qua, và E2/E3 trông y hệt E1.

## 6. Box suy biến làm hỏng cả output

- `score.parse_output` trả `None` cho **cả ảnh** nếu chỉ một box có x2 ≤ x1. Khi làm tròn, box rất nhỏ có thể ra x1 = x2.
- **Xử lý:** `yolo_to_box` nới box ra ít nhất 1 đơn vị. 49 box GT tràn ra ngoài khung được kẹp về [0, 1000].

## 7. Môi trường Colab

- **Colab Secrets:** `SecretNotFoundError` khi chưa tạo secret, hoặc khi gõ giá trị vào ô "Tên". Ô Kaggle giờ hỏi nhập ẩn (`getpass`) nếu thiếu secret.
- **torchao 0.10 có sẵn trên Colab** làm `peft==0.21.2` báo `ImportError` (peft đòi ≥ 0.16). Bài này không dùng torchao, nên ô cài đặt gỡ nó đi. Sau khi gỡ phải khởi động lại phiên.
- **Session bị ngắt:** adapter, dự đoán và `run.json` được ghi thẳng vào Drive. `run_or_resume` bỏ qua thí nghiệm đã xong, và `predict` chạy tiếp từ ảnh đang dở.

## 8. GRPO (TRL 1.9.2)

- TRL tự đổi tham số huấn luyện của model lượng tử sang **bf16**. Với lr 5e-6, phần lớn cập nhật bị làm tròn mất. `run_grpo` đưa chúng về float32 trước khi optimizer được tạo.
- Code viết cho T4 (không hỗ trợ bf16) nên đặt `fp16=True, bf16=False`. Lần chạy thực tế dùng A100 40 GB nhưng vẫn giữ fp16 để cùng cấu hình với SFT. Có 2/60 bước bị GradScaler bỏ qua vì gradient tràn số.
- Cột `gt` (JSON) đi vào reward qua `**kwargs`. Completion có dạng hội thoại `[{"role": "assistant", "content": …}]`.

## 9. Đo VRAM

- `torch.cuda.max_memory_allocated()` tính cả tensor còn sót lại từ các ô trước trong cùng session. VRAM của E1 (9.62 GB) cao hơn E2 (8.04 GB), trong khi chạy thử với cấu hình E1 chỉ đỉnh 6.35 GB. Nếu cần số sạch, hãy chạy mỗi thí nghiệm trong một phiên mới.
