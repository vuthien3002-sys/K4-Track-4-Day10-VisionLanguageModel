import json

from day10.report import image_result, output_stats, summary_rows

ENTRY = {"image_id": "x", "width": 640, "height": 360,
         "objects": [{"label": "pothole", "yolo": [0.15, 0.15, 0.1, 0.1]}]}


def test_image_result_matches_referee_rules():
    ok = json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}])
    assert image_result(ENTRY, ok)[1:] == (1, 0, 0)
    assert image_result(ENTRY, "garbage")[1:] == (0, 0, 1)


def test_output_stats():
    preds = {"x": "[]"}
    s = output_stats([ENTRY], preds)
    assert s["empty_rate"] == 1.0 and s["invalid_rate"] == 0.0 and s["gt_boxes_per_image"] == 1.0


def test_summary_rows(tmp_path):
    ev = {"name": "E1", "micro": {"f1": 0.5}, "f1_ci95": [0.4, 0.6], "fp_per_image": 0.25,
          "per_class": {c: {"f1": 0.1} for c in ("pothole", "crack", "manhole", "other")},
          "meta": {"trainable_params": 123, "peak_vram_gb": 9.1, "train_minutes": 20.5}}
    (tmp_path / "e1.json").write_text(json.dumps(ev), encoding="utf-8")
    (row,) = summary_rows(tmp_path)
    assert row.startswith("| E1 | E1 | 123 | 9.10 | 20.5 | 0.500 | [0.400, 0.600] | 0.25 |")


def test_pick_examples_disjoint():
    from day10.report import pick_examples
    entries = [dict(ENTRY, image_id=f"i{k}") for k in range(4)]
    ok = json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}])
    preds = {"i0": ok, "i1": ok, "i2": "[]", "i3": "bad"}
    good, bad = pick_examples(entries, preds, 3, 3)
    assert {e["image_id"] for e in good} == {"i0", "i1"}
    assert {e["image_id"] for e in bad} == {"i2", "i3"}


def test_paired_bootstrap_identical_and_better():
    from day10.report import paired_bootstrap_f1
    entries = [dict(ENTRY, image_id=f"i{k}") for k in range(20)]
    ok = json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}])
    good = {e["image_id"]: ok for e in entries}
    half = {e["image_id"]: (ok if k % 2 else "[]") for k, e in enumerate(entries)}
    same = paired_bootstrap_f1(entries, good, good, n_boot=200)
    assert same["delta"] == 0 and same["ci95"] == [0, 0]
    up = paired_bootstrap_f1(entries, half, good, n_boot=200)
    assert up["delta"] > 0 and up["ci95"][0] > 0 and up["p_not_better"] == 0
