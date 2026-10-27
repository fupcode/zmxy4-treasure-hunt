import argparse
import math
import time

from zmxy4_treasure_hunt.maze.navigation import Graph, Maze
from zmxy4_treasure_hunt.maze.runner import FindTreasure

__all__ = ["FindTreasure", "Graph", "Maze", "main"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="迷宫自动寻宝（DXcam BGR 截图）")
    parser.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="手动指定主显示器内的游戏区域，默认自动定位")
    parser.add_argument("--interval", type=float, default=0.05, help="最小循环间隔，单位秒，默认 0.05")
    parser.add_argument("--delay", type=float, default=2, help="启动前等待秒数，默认 2")
    args = parser.parse_args(argv)
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay 必须为非负数且有限")
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval 必须为正数且有限")
    try:
        time.sleep(args.delay)
        FindTreasure(region=args.region, interval=args.interval).main()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
