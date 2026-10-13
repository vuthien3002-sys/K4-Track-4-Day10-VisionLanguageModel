import json

from day10.labels import (box_to_pixels, objects_to_answer, prompt_for, session_overlap, split_stats, subset,
                          yolo_to_box, yolo_to_pixels)
from scripts.score import parse_output


def test_yolo_to_box_norm1000():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.1]) == {"label": "pothole", "bbox_2d": [400, 450, 600, 550]}


def test_yolo_to_box_clamps_and_never_degenerate():
    assert yolo_to_box("crack", [0.02, 0.5, 0.1, 0.1])["bbox_2d"][0] == 0
    b = yolo_to_box("crack", [0.5, 0.5, 0.0001, 0.0001])["bbox_2d"]
    assert b[2] > b[0] and b[3] > b[1]
    b = yolo_to_box("crack", [1.0, 1.0, 0.0001, 0.0001])["bbox_2d"]
    assert b == [999, 999, 1000, 1000]


def test_pixel_scale():
    assert yolo_to_box("manhole", [0.5, 0.5, 0.5, 0.5], "pixel", 640, 360)["bbox_2d"] == [160, 90, 480, 270]
    assert "640x360" in prompt_for("pixel", 640, 360) and "0-1000" in prompt_for("norm1000")


def test_answer_sorted_and_parseable_by_referee():
    objs = [{"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]}, {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.5, 0.2, 0.1, 0.1]}]
    ans = objects_to_answer(objs)
    assert [b["label"] for b in json.loads(ans)] == ["pothole", "manhole", "crack"]
    assert " " not in ans
    assert len(parse_output(ans)) == 3
    assert objects_to_answer([]) == "[]"


def test_round_trip_to_pixels_within_rounding():
    yolo = [0.31, 0.62, 0.05, 0.03]
    box = yolo_to_box("pothole", yolo)["bbox_2d"]
    got, want = box_to_pixels(box, 640, 360), yolo_to_pixels(yolo, 640, 360)
    assert all(abs(a - b) <= 0.5 for a, b in zip(got, want))


def test_session_overlap_and_stats():
    tr = [{"session": "a", "objects": [{"label": "crack"}]}, {"session": "b", "objects": []}]
    te = [{"session": "c", "objects": [{"label": "pothole"}, {"label": "crack"}]}]
    assert all(not v for v in session_overlap({"train": tr, "test": te}).values())
    assert session_overlap({"train": tr, "test": tr[:1]})[("test", "train")] == {"a"}
    assert split_stats(te) == (1, {"pothole": 1, "crack": 1, "manhole": 0})


def test_subset_deterministic():
    rows = list(range(1422))
    a, b = subset(rows, 0.1, seed=42), subset(rows, 0.1, seed=42)
    assert a == b and len(a) == 142 and subset(rows, 1.0) == rows
