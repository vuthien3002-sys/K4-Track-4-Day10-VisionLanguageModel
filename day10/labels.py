"""Nhãn grounding, prompt và các kiểm tra dữ liệu. Thuần Python, không cần GPU."""
import json
import random

from scripts.labcommon import CLASSES

PROMPT_NORM1000 = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)

# Chỉ dùng cho E4 coord_scale=pixel: nhãn, prompt và cách chấm đều đổi sang pixel.
PROMPT_PIXEL = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "in absolute pixels of the {width}x{height} image: "
    '[{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}]. '
    "If no damage is found, output []."
)


def prompt_for(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        return PROMPT_NORM1000
    if coord == "pixel":
        return PROMPT_PIXEL.format(width=width, height=height)
    raise ValueError(f"coord không hợp lệ: {coord}")


def _scale(coord, width, height):
    return (1000, 1000) if coord == "norm1000" else (width, height)


def _clamp(v, hi):
    return max(0, min(hi, int(round(v))))


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (tâm + kích thước, chuẩn hoá [0,1]) → {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = _scale(coord, width, height)
    x1, y1 = _clamp((xc - w / 2) * sx, sx), _clamp((yc - h / 2) * sy, sy)
    x2, y2 = _clamp((xc + w / 2) * sx, sx), _clamp((yc + h / 2) * sy, sy)
    # Box rất nhỏ có thể bị làm tròn thành x1 == x2; score.parse_output coi cả output là hỏng
    # nếu có box suy biến, nên phải nới ra ít nhất 1 đơn vị.
    if x2 <= x1:
        x1, x2 = (x1, x1 + 1) if x1 < sx else (sx - 1, sx)
    if y2 <= y1:
        y1, y2 = (y1, y1 + 1) if y1 < sy else (sy - 1, sy)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    """Chuỗi JSON đáp án, box sắp theo (y1, x1): trên → dưới, trái → phải."""
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixels(bbox, width, height, coord="norm1000"):
    """bbox_2d (thang 0–1000 hoặc pixel) → toạ độ pixel thực để vẽ."""
    if coord == "pixel":
        return tuple(float(v) for v in bbox)
    x1, y1, x2, y2 = bbox
    return (x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height)


def yolo_to_pixels(yolo, width, height):
    xc, yc, w, h = yolo
    return ((xc - w / 2) * width, (yc - h / 2) * height, (xc + w / 2) * width, (yc + h / 2) * height)


def split_stats(entries):
    """(số ảnh, {lớp: số box}) của một split."""
    boxes = {c: 0 for c in CLASSES}
    for e in entries:
        for o in e["objects"]:
            boxes[o["label"]] += 1
    return len(entries), boxes


def session_overlap(splits):
    """{(a, b): các session xuất hiện ở cả split a và b}. Đúng thì mọi giá trị đều rỗng."""
    sess = {name: {e["session"] for e in rows} for name, rows in splits.items()}
    names = sorted(sess)
    return {(a, b): sess[a] & sess[b] for i, a in enumerate(names) for b in names[i + 1:]}


def subset(entries, fraction, seed=42):
    """Lấy ngẫu nhiên (có seed) một phần ảnh train cho E4 data_size; giữ thứ tự gốc."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction phải trong (0, 1]")
    if fraction == 1:
        return list(entries)
    k = max(1, round(len(entries) * fraction))
    keep = set(random.Random(seed).sample(range(len(entries)), k))
    return [e for i, e in enumerate(entries) if i in keep]
