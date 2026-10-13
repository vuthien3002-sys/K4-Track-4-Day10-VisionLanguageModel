"""Reward IoU kiểm chứng được cho E5 (GRPO). Thuần Python, test được không cần GPU."""
import json

from scripts.score import IOU_THR, _gt_boxes, iou_xyxy, parse_output


def gt_boxes(entry):
    """GT thang 0–1000, cùng công thức với trọng tài score.py (tránh lệch thang)."""
    return [{"label": g["label"], "bbox": list(g["bbox"])} for g in _gt_boxes(entry)]


def compute_reward(completion, gt):
    """0.0 nếu JSON hỏng; 1.0 nếu cả GT và dự đoán rỗng; ngược lại 0.2 + 0.8·ΣIoU / max(n_gt, n_pred, 1)."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt and not pred:
        return 1.0
    total = 0.0
    for lab in {g["label"] for g in gt} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)),
                       reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total += v
    return 0.2 + 0.8 * total / max(len(gt), len(pred), 1)


def completion_text(completion):
    """TRL trả completion dạng chuỗi hoặc dạng hội thoại [{"role": "assistant", "content": ...}]."""
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def grpo_reward(prompts, completions, gt, **kwargs):
    """Chữ ký reward function của trl.GRPOTrainer: cột `gt` của dataset (chuỗi JSON) đi vào qua kwargs."""
    return [compute_reward(completion_text(c), json.loads(g)) for c, g in zip(completions, gt)]
