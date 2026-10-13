"""Nạp Qwen3-VL-4B 4-bit, LoRA/đóng băng, huấn luyện và suy luận. Cần GPU (Colab T4)."""
import gc
import json
import math
import random
import re
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from PIL import Image

from day10.labels import objects_to_answer, prompt_for
from scripts.labcommon import read_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# transformers so pattern bỏ lượng tử hoá bằng re.match từ ĐẦU tên module. Tên thật là
# "model.visual.blocks.0.attn.qkv", nên "visual" (như trong GUIDE) không khớp và ViT bị lượng tử hoá 4-bit.
SKIP_MODULES = ["model.visual", "lm_head"]

# PEFT dùng re.fullmatch khi target_modules là chuỗi: khớp đúng tên đầy đủ, không nhầm sang projector.
LLM_TARGETS = r"model\.language_model\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")


# ---------------------------------------------------------------- model

def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_MODULES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quantization, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def free():
    """Gọi sau `del model` để trả VRAM giữa các thí nghiệm trong cùng session."""
    gc.collect()
    torch.cuda.empty_cache()


def _block(name):
    if "visual.merger" in name:
        return "projector.merger"
    if "visual.deepstack_merger_list" in name:
        return "projector.deepstack"
    if "visual" in name:
        return "vit"
    return "llm"


def _numel(p):
    # Params4bit lưu 2 trọng số/byte; quant_state.shape là kích thước gốc.
    qs = getattr(p, "quant_state", None)
    return math.prod(qs.shape) if qs is not None else p.numel()


def param_breakdown(model):
    """Số tham số gốc theo khối (Câu 1) và số Linear bị lượng tử hoá 4-bit theo khối."""
    params, linear4 = {}, {}
    for n, p in model.named_parameters():
        params[_block(n)] = params.get(_block(n), 0) + _numel(p)
    for n, m in model.named_modules():
        if type(m).__name__ == "Linear4bit":
            key = "lm_head" if n.endswith("lm_head") else _block(n)
            linear4[key] = linear4.get(key, 0) + 1
    return params, linear4


def check_quantization(model):
    _, linear4 = param_breakdown(model)
    bad = {k: v for k, v in linear4.items() if k != "llm"}
    if bad:
        raise RuntimeError(f"ViT/projector/lm_head bị lượng tử hoá 4-bit: {bad}")
    return linear4


# ---------------------------------------------------------------- dữ liệu huấn luyện

def build_messages(image, prompt, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def encode_example(processor, image, prompt, answer):
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prompt_enc = processor.apply_chat_template(build_messages(image, prompt), add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template(build_messages(image, prompt, answer),
                                             add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: kiểm tra chat template")
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = -100  # chỉ học phần trả lời của assistant
    item = {k: full_enc[k][0] for k in ("input_ids", "attention_mask", "mm_token_type_ids") if k in full_enc}
    item["labels"] = labels[0]
    item["pixel_values"] = full_enc["pixel_values"]
    item["image_grid_thw"] = full_enc["image_grid_thw"]
    return item


def collate(batch, pad_id):
    def pad(key, value):
        return torch.nn.utils.rnn.pad_sequence([b[key] for b in batch], batch_first=True, padding_value=value)

    out = {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
           "labels": pad("labels", -100)}
    if "mm_token_type_ids" in batch[0]:  # transformers 5.x báo lỗi nếu thiếu (M-RoPE)
        out["mm_token_type_ids"] = pad("mm_token_type_ids", 0)
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)  # nối, KHÔNG pad
    out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return out


class GroundingDataset(torch.utils.data.Dataset):
    def __init__(self, entries, processor, data_dir="data", coord="norm1000"):
        self.entries, self.processor, self.data_dir, self.coord = entries, processor, Path(data_dir), coord

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        image = Image.open(self.data_dir / e["image_path"]).convert("RGB")
        prompt = prompt_for(self.coord, e["width"], e["height"])
        answer = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        return encode_example(self.processor, image, prompt, answer)


def token_breakdown(processor, item):
    """Đếm token ảnh / prompt chữ / trả lời của một ví dụ (Bước 4, Câu 4)."""
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    ids, labels = item["input_ids"], item["labels"]
    n_image = int((ids == image_id).sum())
    n_answer = int((labels != -100).sum())
    grid = item["image_grid_thw"][0].tolist()
    return {"total": len(ids), "image": n_image, "answer": n_answer,
            "prompt_text_and_template": len(ids) - n_image - n_answer,
            "image_grid_thw": grid, "image_from_grid": grid[0] * grid[1] * grid[2] // 4,
            "answer_decoded": processor.tokenizer.decode(ids[labels != -100])}


# ---------------------------------------------------------------- setup E1/E2/E3

def load_config(name, configs_dir="configs"):
    return yaml.safe_load((Path(configs_dir) / f"{name.lower()}.yaml").read_text(encoding="utf-8"))


def apply_setup(model, cfg):
    """Đóng băng tất cả, gắn LoRA theo cfg["lora"]["targets"], mở projector nếu cfg["train_projector"]."""
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()  # cần cho gradient checkpointing
    lora = cfg["lora"]
    target = {"llm": LLM_TARGETS, "llm+vit": f"(?:{LLM_TARGETS})|(?:{VIT_TARGETS})"}[lora["targets"]]
    matched = [n for n, _ in model.named_modules() if re.fullmatch(target, n)]
    peft_model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=target, bias="none", task_type="CAUSAL_LM"))
    if cfg["train_projector"]:
        for n, p in peft_model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.to(torch.float32)  # GradScaler không unscale được grad fp16
                p.requires_grad = True
    bad = [n for n, p in peft_model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:
        raise RuntimeError(f"tham số huấn luyện không phải float32: {bad[:3]}")
    return peft_model, matched


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------- lưu / nạp

def save_checkpoint(model, out_dir):
    """peft chỉ lưu adapter LoRA; projector huấn luyện đầy đủ phải lưu riêng."""
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir / "adapter")
    extra = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n}
    if extra:
        torch.save(extra, out_dir / "extra_trainable.pt")
    return len(extra)


def load_trained(out_dir, model_id=MODEL_ID, is_trainable=False):
    """Model gốc 4-bit → adapter LoRA → nạp lại projector (nếu có) và kiểm tra đã nạp đúng."""
    from peft import PeftModel

    out_dir = Path(out_dir)
    model, processor = load_model(model_id)
    model = PeftModel.from_pretrained(model, out_dir / "adapter", is_trainable=is_trainable)
    extra_path = out_dir / "extra_trainable.pt"
    if extra_path.exists():
        extra = torch.load(extra_path, map_location="cpu")
        result = model.load_state_dict(extra, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(f"tên projector không khớp: {result.unexpected_keys[:3]}")
        current = dict(model.named_parameters())
        name = next(iter(extra))
        if not torch.allclose(current[name].detach().float().cpu(), extra[name].float(), atol=1e-2):
            raise RuntimeError(f"projector chưa được nạp: {name}")
        print(f"đã nạp lại {len(extra)} tensor projector từ {extra_path}")
    return model, processor


# ---------------------------------------------------------------- huấn luyện

def train(model, processor, entries, cfg, out_dir, data_dir="data", coord="norm1000", log=print):
    from transformers import get_cosine_schedule_with_warmup

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        GroundingDataset(entries, processor, data_dir, coord), batch_size=cfg["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(seed), drop_last=True, num_workers=2,
        collate_fn=partial(collate, pad_id=processor.tokenizer.pad_token_id))
    params = [p for p in model.parameters() if p.requires_grad]
    max_steps, accum = cfg["max_steps"], cfg["grad_accum"]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, round(0.05 * max_steps)), max_steps)
    scaler = torch.amp.GradScaler("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    device = next(model.parameters()).device

    torch.cuda.reset_peak_memory_stats()
    t0, step, history, it = time.time(), 0, [], iter(loader)
    while step < max_steps:
        step_loss = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step + 1}: {loss.item()} (tràn số fp16?)")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        history.append({"step": step, "loss": step_loss, "lr": sched.get_last_lr()[0], "grad_norm": grad_norm})
        if step == 1 or step % 10 == 0:
            log(f"bước {step}/{max_steps}  loss={step_loss:.4f}  lr={history[-1]['lr']:.2e}  "
                f"{(time.time() - t0) / 60:.1f} phút  VRAM đỉnh={torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
        if step % 100 == 0 or step == max_steps:
            save_checkpoint(model, out_dir)

    run = {"trainable_params": count_trainable(model),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1),
           "n_train_images": len(entries), "coord": coord, "config": cfg, "history": history}
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    return run


# ---------------------------------------------------------------- suy luận

@torch.no_grad()
def predict(model, processor, entries, out_path, data_dir="data", coord="norm1000", batch_size=4,
            max_new_tokens=256, log=print):
    """Ghi {"image_id", "raw"} từng dòng; chạy lại sẽ bỏ qua ảnh đã có (an toàn khi Colab bị ngắt)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [e for e in entries if e["image_id"] not in done]
    model.eval()
    if getattr(model, "is_gradient_checkpointing", False):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            msgs = [build_messages(Image.open(Path(data_dir) / e["image_path"]).convert("RGB"),
                                   prompt_for(coord, e["width"], e["height"])) for e in chunk]
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt",
                                                   processor_kwargs={"padding": True}).to(device)
            # autocast: projector đã huấn luyện ở float32, phần còn lại fp16
            with torch.autocast("cuda", dtype=torch.float16):
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, t in zip(chunk, texts):
                f.write(json.dumps({"image_id": e["image_id"], "raw": t}, ensure_ascii=False) + "\n")
            f.flush()
            n = min(i + batch_size, len(todo))
            if n == len(todo) or (i // batch_size) % 10 == 0:
                log(f"{len(done) + n}/{len(entries)} ảnh  {(time.time() - t0) / 60:.1f} phút")
    return out_path


# ---------------------------------------------------------------- một thí nghiệm trọn gói

EXPECTED_LORA_MODULES = {"llm": 252, "llm+vit": 348}


def run_experiment(name, cfg, train_entries, test_entries, work_dir, data_dir="data", coord="norm1000",
                   log=print):
    """Nạp model gốc → setup → train 300 bước → lưu → suy luận test → work_dir/<name>.jsonl."""
    work_dir = Path(work_dir)
    model, processor = load_model()
    check_quantization(model)
    model, matched = apply_setup(model, cfg)
    expected = EXPECTED_LORA_MODULES[cfg["lora"]["targets"]]
    if len(matched) != expected:
        raise RuntimeError(f"{name}: {len(matched)} module LoRA, kỳ vọng {expected}")
    log(f"{name}: {len(matched)} module LoRA, {count_trainable(model):,} tham số huấn luyện")
    run = train(model, processor, train_entries, cfg, work_dir / name.lower(), data_dir, coord, log)
    log(f"{name}: train xong {run['train_minutes']} phút, VRAM đỉnh {run['peak_vram_gb']} GB")
    pred_path = predict(model, processor, test_entries, work_dir / f"{name.lower()}.jsonl", data_dir, coord,
                        log=log)
    del model
    free()
    return run, pred_path


def run_or_resume(name, cfg, train_entries, test_entries, work_dir, data_dir="data", coord="norm1000", log=print):
    """Như run_experiment, nhưng nếu đã train xong (có run.json) thì chỉ nạp lại và suy luận nốt phần còn thiếu."""
    work_dir = Path(work_dir)
    run_path = work_dir / name.lower() / "run.json"
    pred_path = work_dir / f"{name.lower()}.jsonl"
    if not run_path.exists():
        return run_experiment(name, cfg, train_entries, test_entries, work_dir, data_dir, coord, log)
    run = json.loads(run_path.read_text(encoding="utf-8"))
    n_done = len(read_jsonl(pred_path)) if pred_path.exists() else 0
    if n_done < len(test_entries):
        log(f"{name}: đã train, nạp lại adapter và suy luận tiếp từ {n_done}/{len(test_entries)} ảnh")
        model, processor = load_trained(work_dir / name.lower())
        predict(model, processor, test_entries, pred_path, data_dir, coord, log=log)
        del model
        free()
    else:
        log(f"{name}: đã có đủ dự đoán, dùng lại {pred_path}")
    return run, pred_path
