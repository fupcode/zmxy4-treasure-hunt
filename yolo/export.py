"""将训练权重导出为 YOLO26 免 NMS ONNX。"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path,
        default=Path(__file__).resolve().parent / "runs/zaomeng/weights/best.pt",
    )
    parser.add_argument("--max-det", type=int, default=32, help="检测数量上限，密集子弹场景可调大")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        parser.error(f"权重不存在：{weights}")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # end2end 表示当前选中的推理分支，不代表是否保留了免 NMS 检测头。
    # nms=False 会在导出时切换到 one-to-one 分支。
    head = model.model.model[-1]
    if any(getattr(head, name, None) is None for name in ("one2one_cv2", "one2one_cv3")):
        parser.error("需要支持免 NMS 检测头的 YOLO26 权重")
    output = model.export(
        format="onnx", imgsz=(512, 800), batch=1, dynamic=False,
        nms=False, max_det=args.max_det, device=args.device,
    )
    print(f"ONNX 已导出：{output}")


if __name__ == "__main__":
    main()
