"""将训练权重导出为 YOLO26 免 NMS ONNX。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path,
        default=Path(__file__).resolve().parent / "runs/zaomeng/weights/best.pt",
    )
    parser.add_argument("--max-det", type=int, default=300, help="检测数量上限，密集子弹场景可调大")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        parser.error(f"权重不存在：{weights}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if not getattr(model.model, "end2end", False):
        parser.error("需要支持免 NMS 检测头的 YOLO26 权重")
    output = model.export(
        format="onnx", imgsz=(512, 800), batch=1, dynamic=False,
        nms=False, max_det=args.max_det, device=args.device,
    )
    print(f"ONNX 已导出：{output}")


if __name__ == "__main__":
    main()
