"""训练造梦航海目标检测模型，运行示例见同目录 README.md。"""

import argparse
from multiprocessing import freeze_support
from pathlib import Path

import yaml


YOLO_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=YOLO_DIR / "zaomeng.v2i.yolo26" / "data.yaml",
        help="数据集 YAML，默认使用随项目准备的数据集",
    )
    parser.add_argument("--model", default="yolo26n.pt", help="预训练模型名称或本地权重路径")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=800, help="训练长边尺寸，配合矩形训练")
    parser.add_argument("--batch", type=int, default=16, help="批大小；显存不足时调小")
    parser.add_argument("--device", default=None, help="例如 0、0,1 或 cpu；默认自动选择")
    parser.add_argument("--workers", type=int, default=0, help="数据加载进程数，Windows 默认 0")
    parser.add_argument("--patience", type=int, default=30, help="连续多少轮无提升后提前停止")
    parser.add_argument("--project", type=Path, default=YOLO_DIR / "runs", help="输出根目录")
    parser.add_argument("--name", default="zaomeng", help="本次训练名称，重名自动递增")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--resume", type=Path, help="从中断训练的 last.pt 恢复原训练计划")
    parser.add_argument("--dry-run", action="store_true", help="检查数据集路径并显示参数，不启动训练")
    return parser.parse_args()


def check_dataset(data_file: Path) -> None:
    """提前检查必需分集；相对路径按数据集 YAML 所在目录解析。"""
    with data_file.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"数据集配置必须是 YAML 字典：{data_file}")
    root = Path(config.get("path") or data_file.parent)
    if not root.is_absolute():
        raise ValueError("自定义 data.yaml 的 path 请使用绝对路径，或省略 path")
    for split in ("train", "val"):
        entries = config.get(split)
        if not entries:
            raise ValueError(f"数据集配置缺少 {split}：{data_file}")
        for entry in entries if isinstance(entries, list) else [entries]:
            split_path = (root / entry).resolve()
            if not split_path.exists():
                raise FileNotFoundError(f"{split} 路径不存在：{split_path}")


def main() -> None:
    args = parse_args()
    if args.resume:
        checkpoint = args.resume.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"断点权重不存在：{checkpoint}")
        if args.dry_run:
            print(f"将从断点恢复原训练计划：{checkpoint}")
            return
        from ultralytics import YOLO

        # 恢复时沿用 checkpoint 的参数，只覆盖设备和加载进程数。
        overrides = {"workers": args.workers}
        if args.device is not None:
            overrides["device"] = args.device
        YOLO(str(checkpoint)).train(resume=True, **overrides)
        return

    data_file = args.data.resolve()
    check_dataset(data_file)
    options = {
        "data": str(data_file),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "rect": True,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(args.project.resolve()),
        "name": args.name,
        "seed": args.seed,
    }
    if args.device is not None:
        options["device"] = args.device
    print(f"模型：{args.model}")
    print(yaml.safe_dump(options, allow_unicode=True, sort_keys=False))
    if args.dry_run:
        print("数据集路径检查通过，未启动训练。")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(**options)
    print(f"训练结果：{model.trainer.save_dir}")
    print(f"最佳权重：{model.trainer.best}")


if __name__ == "__main__":
    freeze_support()
    main()
