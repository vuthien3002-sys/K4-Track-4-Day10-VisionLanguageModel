"""Các phân tích dùng trong REPORT.md (không cần GPU).

Chạy: python -m day10.analysis --preds <thư mục chứa e0.jsonl…e5.jsonl> --grpo-log report/e5_grpo_log.json
"""
import argparse
import json
import random
import re
import statistics
from pathlib import Path

from day10.report import load_preds, output_stats, paired_bootstrap_f1
from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, iou_xyxy, parse_output

NAMES = ("e0", "e1", "e2", "e3", "e4_data_0.1", "e5")
PAIRS = (("e0", "e1"), ("e1", "e2"), ("e2", "e3"), ("e1", "e3"), ("e4_data_0.1", "e3"), ("e1", "e4_data_0.1"),
         ("e3", "e5"))


def _ts(name):
    m = re.search(r"(\d+)h(\d+)m(\d+)s(\d+)", name)
    return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000 if m else None


def frame_similarity(rows, data_dir="data", n=150, seed=0):
    """Khoảng cách thời gian giữa frame liền kề trong một phiên và độ khác biệt pixel so với cặp khác phiên."""
    import numpy as np
    from PIL import Image

    by = {}
    for r in rows:
        t = _ts(r["image_id"])
        if t is not None:
            by.setdefault(r["session"], []).append((t, r))
    gaps, pairs = [], []
    for xs in by.values():
        xs.sort(key=lambda x: x[0])
        for a, b in zip(xs, xs[1:]):
            gaps.append(b[0] - a[0])
            pairs.append((a[1], b[1]))

    def arr(r):
        return np.asarray(Image.open(Path(data_dir) / r["image_path"]).convert("L").resize((64, 36)), dtype=np.float32)

    rng = random.Random(seed)
    near = [p for p, g in zip(pairs, gaps) if g <= 5]
    same = [float(np.abs(arr(a) - arr(b)).mean()) for a, b in rng.sample(near, min(n, len(near)))]
    cross = []
    while len(cross) < n:
        a, b = rng.sample(rows, 2)
        if a["session"] != b["session"]:
            cross.append(float(np.abs(arr(a) - arr(b)).mean()))
    return {"frames_with_time": sum(len(v) for v in by.values()), "gap_median_s": statistics.median(gaps),
            "pairs_within_5s": len(near), "diff_consecutive": statistics.median(same),
            "diff_other_session": statistics.median(cross)}


def recall_by_size(entries, preds):
    """Recall@IoU0.5 của từng GT box, chia ba nhóm diện tích bằng nhau."""
    gts = []
    for e in entries:
        pred = parse_output(preds.get(e["image_id"])) or []
        for g in _gt_boxes(e):
            best = max((iou_xyxy(g["bbox"], p["bbox"]) for p in pred if p["label"] == g["label"]), default=0)
            b = g["bbox"]
            gts.append(((b[2] - b[0]) * (b[3] - b[1]) / 1e6, g["label"], best >= 0.5))
    gts.sort()
    k = len(gts) // 3
    out = {}
    for name, chunk in (("small", gts[:k]), ("medium", gts[k:2 * k]), ("large", gts[2 * k:])):
        out[name] = {"area": [chunk[0][0], chunk[-1][0]], "recall": sum(x[2] for x in chunk) / len(chunk)}
    for c in CLASSES:
        sub = [x for x in gts if x[1] == c]
        out[c] = {"recall": sum(x[2] for x in sub) / len(sub), "small_share": sum(x[0] < gts[k][0] for x in sub) / len(sub)}
    return out


def error_types(entries, preds):
    """Dự đoán không khớp: lệch vị trí (cùng lớp, 0.1 ≤ IoU < 0.5), nhầm lớp (IoU ≥ 0.5 với lớp khác), bịa trên nền."""
    loc = cls = bg = 0
    for e in entries:
        gt = _gt_boxes(e)
        for p in parse_output(preds.get(e["image_id"])) or []:
            same = max((iou_xyxy(p["bbox"], g["bbox"]) for g in gt if g["label"] == p["label"]), default=0)
            other = max((iou_xyxy(p["bbox"], g["bbox"]) for g in gt if g["label"] != p["label"]), default=0)
            if same >= 0.5:
                continue
            if same >= 0.1:
                loc += 1
            elif other >= 0.5:
                cls += 1
            else:
                bg += 1
    return {"mislocalised": loc, "wrong_class": cls, "hallucinated": bg}


def grpo_summary(logs):
    steps = [x for x in logs if "reward" in x]

    def avg(key, part):
        xs = [x[key] for x in part if isinstance(x.get(key), (int, float)) and x[key] == x[key]]
        return sum(xs) / len(xs) if xs else None

    keys = ("reward", "reward_std", "frac_reward_zero_std", "completions/mean_length", "completions/clipped_ratio")
    return {"steps": len(steps), "skipped_nan_grad": [x["step"] for x in steps if x.get("grad_norm") != x.get("grad_norm")],
            "first10": {k: avg(k, steps[:10]) for k in keys}, "last10": {k: avg(k, steps[-10:]) for k in keys},
            "runtime_min": logs[-1].get("train_runtime", 0) / 60}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", default="work/preds")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--grpo-log", default="report/e5_grpo_log.json")
    a = ap.parse_args(argv)
    splits = {s: read_jsonl(Path(a.splits_dir) / f"{s}.jsonl") for s in ("train", "val", "test")}
    test = splits["test"]
    preds = {n: load_preds(Path(a.preds) / f"{n}.jsonl") for n in NAMES if (Path(a.preds) / f"{n}.jsonl").exists()}

    print("== session theo split")
    for s, rows in splits.items():
        print(f"{s:5s} sessions={len({r['session'] for r in rows})}")
    print("== frame liền kề:", frame_similarity([r for rows in splits.values() for r in rows]))
    print("== bootstrap ghép cặp ΔF1 = F1(B) − F1(A)")
    for x, y in PAIRS:
        if x in preds and y in preds:
            r = paired_bootstrap_f1(test, preds[x], preds[y])
            print(f"{y} vs {x}: Δ={r['delta']:+.3f} CI95=[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
                  f"P(Δ≤0)={r['p_not_better']:.3f}")
    print("== thống kê output")
    for n, p in preds.items():
        print(n, {k: round(v, 3) for k, v in output_stats(test, p).items()})
    print("== diện tích box (tỷ lệ khung hình), trung vị")
    for c in CLASSES:
        tr = [o["yolo"][2] * o["yolo"][3] for r in splits["train"] for o in r["objects"] if o["label"] == c]
        te = [o["yolo"][2] * o["yolo"][3] for r in test for o in r["objects"] if o["label"] == c]
        print(f"{c:8s} train n={len(tr)} {statistics.median(tr):.2%}  test n={len(te)} {statistics.median(te):.2%}")
    if "e3" in preds:
        print("== E3 recall theo kích thước:", json.dumps(recall_by_size(test, preds["e3"]), ensure_ascii=False))
        print("== E3 loại lỗi:", error_types(test, preds["e3"]))
    if Path(a.grpo_log).exists():
        print("== GRPO:", json.dumps(grpo_summary(json.loads(Path(a.grpo_log).read_text(encoding="utf-8"))),
                                     ensure_ascii=False))


if __name__ == "__main__":
    main()
