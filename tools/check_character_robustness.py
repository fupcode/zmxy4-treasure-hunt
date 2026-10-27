"""离线测试人物黑圈识别：截图扰动 → 遮罩反算 → 当前人物识别。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zmxy4_treasure_hunt.maze.vision.mask import MaskRestorer
from zmxy4_treasure_hunt.maze.vision import DetectionError, locate_character

# 依据原图人工圈定人物，坐标与识别器输出相互独立。
ANNOTATIONS = {
    "ch2": {"center": (154, 143), "box": (114, 94, 194, 184)},
    "ch5": {"center": (155, 143), "box": (114, 94, 194, 184)},
    "ch6": {"center": (143, 148), "box": (101, 106, 184, 190)},
    "ch7": {"center": (146, 146), "box": (104, 96, 188, 190)},
}


def variants(image):
    yield "original", "original", image, 0, 0
    for scale in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.125, 1.25, 1.5, 1.75, 2.0):
        yield "scale", f"scale_{scale}", cv2.resize(image, None, fx=scale, fy=scale), 0, 0
    for name, interpolation in (("nearest", cv2.INTER_NEAREST), ("area", cv2.INTER_AREA), ("cubic", cv2.INTER_CUBIC)):
        for scale in (0.5, 0.75, 1.5):
            yield f"scale_{name}", f"scale_{name}_{scale}", cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation), 0, 0
    for gain in (0.65, 0.85, 1.15, 1.35):
        yield "brightness", f"gain_{gain}", np.clip(image.astype(float) * gain, 0, 255).astype(np.uint8), 0, 0
    for offset in (1, 2, 5):
        yield "black_level", f"black_plus_{offset}", np.clip(image.astype(float) + offset, 0, 255).astype(np.uint8), 0, 0
    for gamma in (0.8, 1.2):
        lut = np.rint((np.arange(256) / 255) ** gamma * 255).astype(np.uint8)
        yield "gamma", f"gamma_{gamma}", cv2.LUT(image, lut), 0, 0
    for quality in (95, 85, 70, 50):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise OSError("JPEG 编码失败")
        yield "jpeg", f"jpeg_{quality}", cv2.imdecode(encoded, cv2.IMREAD_COLOR), 0, 0
    for sigma in (1, 3):
        rng = np.random.default_rng(20261028)
        noise = rng.normal(0, sigma, image.shape)
        yield "noise", f"noise_{sigma}", np.clip(np.rint(image.astype(float) + noise), 0, 255).astype(np.uint8), 0, 0
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        shifted = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                                 (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        yield "alignment", f"shift_{dx}_{dy}", shifted, dx, dy


def evaluate(character, annotation, sx, sy, dx, dy):
    x = (character.x + 0.5) / sx - 0.5 - dx
    y = (character.y + 0.5) / sy - 0.5 - dy
    radius = character.radius / ((sx + sy) / 2)
    error = float(np.linalg.norm(np.array((x, y)) - annotation["center"]))
    left, top, right, bottom = annotation["box"]
    yy, xx = np.mgrid[top:bottom, left:right]
    coverage = float(np.mean((xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2))
    radius_limit = 1.5 * max(right - left, bottom - top)
    passed = error <= 10 and coverage >= 0.95 and radius <= radius_limit
    return {"normalized_center": [round(x, 2), round(y, 2)], "normalized_radius": round(radius, 2),
            "center_error_px": round(error, 2), "portrait_box_coverage": round(coverage, 4),
            "passed": bool(passed)}


def save_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(path)
    path.write_bytes(encoded.tobytes())


def annotate(image, annotation, result):
    marked = image.copy()
    left, top, right, bottom = annotation["box"]
    cv2.rectangle(marked, (left, top), (right, bottom), (255, 200, 0), 1)
    cv2.drawMarker(marked, annotation["center"], (255, 200, 0), cv2.MARKER_CROSS, 10, 1)
    if result.get("found"):
        x, y = (round(value) for value in result["normalized_center"])
        cv2.circle(marked, (x, y), max(1, round(result["normalized_radius"])), (0, 0, 255), 2)
        cv2.circle(marked, (x, y), 3, (0, 0, 255), -1)
        if result['detected'].get('ring_radius') is not None:
            cv2.circle(marked, (x, y), round(result['detected']['ring_radius']), (0, 255, 0), 1)
    return marked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=ROOT / "images/maze")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/maze-character-check")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, annotation in ANNOTATIONS.items():
        image = cv2.imdecode(np.frombuffer((args.image_dir / f"{name}.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            parser.error(f"无法读取 {name}.png")
        for group, label, observed, dx, dy in variants(image):
            frame = MaskRestorer(observed.shape[1], observed.shape[0]).restore(observed)
            result = {"character": name, "group": group, "variant": label, "found": False, "passed": False}
            try:
                character = locate_character(frame.image, frame.valid_pixels)
                result.update(found=True, detected=asdict(character))
                result.update(evaluate(character, annotation, observed.shape[1] / image.shape[1],
                                       observed.shape[0] / image.shape[0], dx, dy))
            except DetectionError as exc:
                result["error"] = str(exc)
            results.append(result)
            if label == "original":
                save_image(args.output_dir / f"{name}_restored.png", frame.image)
                save_image(args.output_dir / f"{name}_detected.png", annotate(frame.image, annotation, result))
        subset = [item for item in results if item["character"] == name]
        print(f"{name}: 合格 {sum(item['passed'] for item in subset)}/{len(subset)}，"
              f"漏检 {sum(not item['found'] for item in subset)}，误定位/掩膜不合格 {sum(item['found'] and not item['passed'] for item in subset)}")
    report = {"criteria": {"center_tolerance_source_px": 10, "minimum_box_coverage": 0.95,
                           "maximum_radius": "1.5 * portrait box max dimension"},
              "annotations": ANNOTATIONS, "results": results,
              "note": "人工人物标注；先扰动截图再反算遮罩。预设用例通过率不等于实机成功率。"}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    groups = list(dict.fromkeys(item["group"] for item in results))
    lines = ["# 人物黑圈识别鲁棒性测试", "", "人工标注中心误差不超过 10 源图像素，人物框覆盖至少 95%，且半径不过大才计为通过。",
             "蓝框/十字为人工参考，绿圈为检测到的黑圈，红圈为人物排除掩膜。", "",
             "| 扰动 | " + " | ".join(f"{name} 通过" for name in ANNOTATIONS) + " |",
             "| --- | " + " | ".join("---" for name in ANNOTATIONS) + " |"]
    for group in groups:
        cells = []
        for name in ANNOTATIONS:
            subset = [item for item in results if item["character"] == name and item["group"] == group]
            cells.append(f"{sum(item['passed'] for item in subset)}/{len(subset)}")
        lines.append(f"| {group} | {' | '.join(cells)} |")
    lines += ["", "## 原图检测", ""]
    for name in ANNOTATIONS:
        baseline = next(item for item in results if item["character"] == name and item["variant"] == "original")
        lines.append(f"- {name}: {json.dumps(baseline, ensure_ascii=False)}")
    lines += ["", "结果只描述这些截图和确定性扰动；没有覆盖所有动画帧、遮挡和人物位置。", "定位使用可见黑色圆弧；缺少足够圆弧支持时仍会拒绝识别。"]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告：{args.output_dir.resolve()}")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
