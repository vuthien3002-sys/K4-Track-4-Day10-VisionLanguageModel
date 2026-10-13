import json

import pytest

from day10.reward import completion_text, compute_reward, grpo_reward, gt_boxes

GT = [{"label": "pothole", "bbox": [100, 100, 200, 200]}]


def _ans(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(b)} for lab, b in boxes])


def test_perfect():
    assert compute_reward(_ans(("pothole", (100, 100, 200, 200))), GT) == pytest.approx(1.0)


def test_broken_json():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [100, 100', GT) == 0.0
    assert compute_reward("no damage", GT) == 0.0


def test_wrong_label():
    assert compute_reward(_ans(("crack", (100, 100, 200, 200))), GT) == pytest.approx(0.2)


def test_extra_box_penalised_by_denominator():
    got = compute_reward(_ans(("pothole", (100, 100, 200, 200)), ("pothole", (500, 500, 600, 600))), GT)
    assert got == pytest.approx(0.6)


def test_both_empty():
    assert compute_reward("[]", []) == 1.0


def test_empty_prediction_only_gets_floor():
    assert compute_reward("[]", GT) == pytest.approx(0.2)


def test_giant_box_gets_floor():
    assert compute_reward(_ans(("pothole", (0, 0, 1000, 1000))), GT) == pytest.approx(0.2)


def test_partial_iou_below_threshold_not_counted():
    # IoU([100,100,200,200], [150,100,250,200]) = 1/3 < 0.5
    assert compute_reward(_ans(("pothole", (150, 100, 250, 200))), GT) == pytest.approx(0.2)


def test_partial_iou_above_threshold_counted():
    # IoU([100,100,200,200], [110,100,210,200]) = 90/110
    assert compute_reward(_ans(("pothole", (110, 100, 210, 200))), GT) == pytest.approx(0.2 + 0.8 * 90 / 110)


def test_degenerate_box_is_invalid():
    assert compute_reward(_ans(("pothole", (200, 100, 100, 200))), GT) == 0.0


def test_gt_boxes_same_scale_as_referee():
    entry = {"objects": [{"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.1]}]}
    assert gt_boxes(entry) == [{"label": "crack", "bbox": [400.0, 450.0, 600.0, 550.0]}]


def test_trl_conversational_completions():
    ans = _ans(("pothole", (100, 100, 200, 200)))
    assert completion_text([{"role": "assistant", "content": ans}]) == ans
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": ans}]}]) == ans
    out = grpo_reward(prompts=[None, None], completions=[[{"role": "assistant", "content": ans}], "[]"],
                      gt=[json.dumps(GT), json.dumps(GT)], trainer_state=None)
    assert out == pytest.approx([1.0, 0.2])
