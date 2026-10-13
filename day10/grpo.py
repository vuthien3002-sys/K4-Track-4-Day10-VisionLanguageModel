"""E5 (tuỳ chọn): GRPO với reward IoU, khởi tạo từ adapter tốt nhất. Cần GPU và trl==1.9.2."""
import json
import random
import shutil
from pathlib import Path

import torch

from day10.labels import prompt_for
from day10.reward import grpo_reward, gt_boxes
from day10.vlm import load_trained


def build_dataset(entries, data_dir="data", n=None, seed=42):
    """Cột `prompt` dạng hội thoại (chuỗi), `image` (TRL tự chèn ảnh vào lượt user), `gt` (JSON cho reward)."""
    from datasets import Dataset, Image as HFImage

    rows = list(entries)
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    data = [{"prompt": [{"role": "user", "content": prompt_for("norm1000")}],
             "image": str(Path(data_dir) / e["image_path"]),
             "gt": json.dumps(gt_boxes(e))} for e in rows]
    return Dataset.from_list(data).cast_column("image", HFImage())


def run_grpo(init_dir, train_entries, out_dir, cfg, data_dir="data", batch_size=4, grad_accum=2, log=print):
    from trl import GRPOConfig, GRPOTrainer

    g = cfg["grpo"]
    out_dir = Path(out_dir)
    model, processor = load_trained(init_dir, is_trainable=True)
    # Projector (nếu có) giữ nguyên như SFT; GRPO chỉ cập nhật LoRA.
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    n_prompts = g["max_steps"] * batch_size * grad_accum // g["num_generations"]
    dataset = build_dataset(train_entries, data_dir, n=n_prompts)
    args = GRPOConfig(
        output_dir=str(out_dir / "trainer"), max_steps=g["max_steps"], learning_rate=g["learning_rate"],
        beta=g["beta"], num_generations=g["num_generations"], max_completion_length=g["max_completion_length"],
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        fp16=True, bf16=False,  # T4 không có bf16
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none", remove_unused_columns=False,
        use_vllm=False, seed=42)
    trainer = GRPOTrainer(model=model, reward_funcs=grpo_reward, args=args, train_dataset=dataset,
                          processing_class=processor)
    # TRL đổi tham số huấn luyện của model lượng tử sang bf16: với lr 5e-6, bước cập nhật nhỏ hơn
    # độ phân giải bf16 nên bị làm tròn mất. Đưa về float32 trước khi optimizer được tạo (trong train()).
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    model.save_pretrained(out_dir / "adapter")
    extra = Path(init_dir) / "extra_trainable.pt"
    if extra.exists():  # để vlm.load_trained(out_dir) nạp lại đúng projector của SFT
        shutil.copy2(extra, out_dir / "extra_trainable.pt")
    logs = trainer.state.log_history
    (out_dir / "grpo_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"GRPO xong, VRAM đỉnh {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, log: {out_dir / 'grpo_log.json'}")
    return model, processor, logs
