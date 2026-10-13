"""Ảnh minh hoạ, thống kê output và bảng tổng hợp cho REPORT.md. Không cần GPU."""
import json
from pathlib import Path

from PIL import Image, ImageDraw

from day10.labels import box_to_pixels, yolo_to_pixels
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output

GT_COLOR = (0, 200, 0)
PRED_COLOR = (230, 30, 30)


def load_preds(path):
    return {r["image_id"]: r["raw"] for r in read_jsonl(path)}


def image_result(entry, raw, coord="norm1000"):
    """(pred đã parse hoặc None, tp, fp, fn) của một ảnh, cùng luật với score.py."""
    parsed = parse_output(raw)
    pred = _to_norm1000(parsed, coord, entry["width"], entry["height"]) if parsed is not None else []
    per, _ = _match(_gt_boxes(entry), pred)
    tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
    return parsed, tp, fp, fn


def draw(entry, raw=None, data_dir="data", coord="norm1000", caption=None):
    """Box đúng màu xanh lá, box dự đoán màu đỏ, chú thích ASCII (font mặc định không có dấu) ở dải trên."""
    img = Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + 22), (255, 255, 255))
    canvas.paste(img, (0, 22))
    d = ImageDraw.Draw(canvas)
    for o in entry["objects"]:
        x1, y1, x2, y2 = yolo_to_pixels(o["yolo"], w, h)
        d.rectangle([x1, y1 + 22, x2, y2 + 22], outline=GT_COLOR, width=3)
        d.text((x1 + 2, y2 + 22 - 11), o["label"], fill=GT_COLOR)
    if raw is not None:
        parsed, tp, fp, fn = image_result(entry, raw, coord)
        for p in parsed or []:
            x1, y1, x2, y2 = box_to_pixels(p["bbox"], w, h, coord)
            d.rectangle([x1, y1 + 22, x2, y2 + 22], outline=PRED_COLOR, width=2)
            d.text((x1 + 2, y1 + 24), p["label"], fill=PRED_COLOR)
        if caption is None:
            status = "invalid JSON" if parsed is None else f"TP={tp} FP={fp} FN={fn}"
            caption = f"{entry['image_id']} | GT(green)={len(entry['objects'])} pred(red)={len(parsed or [])} | {status}"
    d.text((4, 5), caption or f"{entry['image_id']} | GT={len(entry['objects'])}", fill=(0, 0, 0))
    return canvas


def pick_examples(entries, preds, k_good=3, k_bad=3, coord="norm1000"):
    """k_good ảnh đúng nhiều nhất (ít lỗi) và k_bad ảnh sai nhiều nhất, chỉ xét ảnh có GT."""
    scored = []
    for e in entries:
        if e["objects"] and e["image_id"] in preds:
            _, tp, fp, fn = image_result(e, preds[e["image_id"]], coord)
            scored.append((tp, fp + fn, e))
    good = sorted((s for s in scored if s[0] > 0), key=lambda s: (s[1], -s[0]))[:k_good]
    chosen = {id(s[2]) for s in good}
    bad = sorted((s for s in scored if id(s[2]) not in chosen), key=lambda s: (-s[1], s[0]))[:k_bad]
    return [s[2] for s in good], [s[2] for s in bad]


def grid(images, cols=2):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for i, im in enumerate(images):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    return sheet


def output_stats(entries, preds, coord="norm1000"):
    """Dấu hiệu reward hacking / lỗi định dạng: tỷ lệ '[]', số box/ảnh, diện tích box trung bình."""
    n = len(entries)
    empty = invalid = boxes = 0
    areas = []
    for e in entries:
        parsed = parse_output(preds.get(e["image_id"]))
        if parsed is None:
            invalid += 1
            continue
        pred = _to_norm1000(parsed, coord, e["width"], e["height"])
        empty += not pred
        boxes += len(pred)
        areas += [(b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) / 1e6 for b in pred]
    return {"n": n, "invalid_rate": invalid / n, "empty_rate": empty / n,
            "gt_empty_rate": sum(not e["objects"] for e in entries) / n,
            "boxes_per_image": boxes / n, "gt_boxes_per_image": sum(len(e["objects"]) for e in entries) / n,
            "mean_box_area_frac": sum(areas) / len(areas) if areas else 0.0}


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def summary_rows(evidence_dir="report/evidence"):
    """Một dòng markdown cho mỗi file evidence, đúng thứ tự cột của bảng trong REPORT.md."""
    rows = []
    for p in sorted(Path(evidence_dir).glob("e*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m, pc = d.get("meta", {}), d["per_class"]
        ci = d["f1_ci95"]
        rows.append("| " + " | ".join([
            p.stem.upper(), d["name"], _fmt(m.get("trainable_params", 0)), _fmt(m.get("peak_vram_gb", ""), 2),
            _fmt(m.get("train_minutes", ""), 1), _fmt(d["micro"]["f1"]), f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            _fmt(d["fp_per_image"], 2), *(_fmt(pc[c]["f1"]) for c in CLASSES)]) + " |")
    return rows
