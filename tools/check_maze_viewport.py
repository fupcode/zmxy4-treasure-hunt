"""离线检查迷宫视区定位，输出角点/边框标注图、裁剪图与评分。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zmxy4_treasure_hunt.maze.viewport import locate_viewport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=ROOT / "images/maze/full.png")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "images/maze/viewport_debug")
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"无法读取图片：{args.image}")
    match = locate_viewport(image)
    left, top, right, bottom = match.box
    marked = image.copy()
    cv2.rectangle(marked, (left, top), (right - 1, bottom - 1), (0, 255, 0), 2)
    for x, y in ((left, top), (right - 1, top), (right - 1, bottom - 1), (left, bottom - 1)):
        cv2.circle(marked, (x, y), 6, (0, 0, 255), 2)
    label = f"viewport: ({left},{top})-({right},{bottom})  score={match.confidence:.3f}"
    cv2.putText(marked, label, (left, max(25, top - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(marked, label, (left, max(25, top - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "marked.png"), marked)
    cv2.imwrite(str(args.output_dir / "viewport.png"), image[top:bottom, left:right])
    report = asdict(match)
    report["note"] = "confidence 为联合规则评分，并非统计成功概率；corner_support 为角点支持度。"
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    (args.output_dir / "report.json").write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
