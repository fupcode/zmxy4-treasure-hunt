"""RGB 图片检测与可视化：uv run python tests/onnx_test.py images/example.png。"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from zmxy4_treasure_hunt.voyage.core.onnx_model import OnnxModel


PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="待检测图片路径，相对当前工作目录解析")
    parser.add_argument(
        "--model", type=Path, default=PROJECT_DIR / "weights/best.onnx",
        help="ONNX 模型路径，默认项目下 weights/best.onnx",
    )
    parser.add_argument("--conf", type=float, default=0.4, help="置信度阈值，范围 0～1")
    parser.add_argument("--save", type=Path, help="另存检测结果图片")
    parser.add_argument("--no-show", action="store_true", help="不打开窗口，可配合 --save 使用")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf 必须在 0～1 之间")
    if not args.image.is_file():
        parser.error(f"图片不存在：{args.image}")
    if not args.model.is_file():
        parser.error(f"模型不存在：{args.model}")

    try:
        with Image.open(args.image) as source:
            image = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        parser.error(f"无法读取图片：{exc}")

    model = OnnxModel(str(args.model))
    start = perf_counter()
    results = model(image, conf_threshold=args.conf)
    elapsed_ms = (perf_counter() - start) * 1000
    print(f"Detection time: {elapsed_ms:.2f} ms (preprocess + inference + postprocess)")

    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    colors = plt.get_cmap("tab10")
    total = 0
    for index, (class_name, detections) in enumerate(results.items()):
        color = colors(index % 10)
        print(f"{class_name}: {len(detections)}")
        for result in detections:
            total += 1
            x1, y1, x2, y2 = result.xyxyn
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
            ax.add_patch(Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=1.5,
            ))
            ax.text(
                x1, y1, f"{class_name} {result.conf:.2f}", fontsize=9,
                color="white", verticalalignment="bottom",
                bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.8, "pad": 2},
            )

    ax.set_title(f"{args.image.name} | {total} detections | {elapsed_ms:.1f} ms")
    ax.axis("off")
    fig.tight_layout()
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
