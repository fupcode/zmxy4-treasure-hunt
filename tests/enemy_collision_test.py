"""手动图片测试：python tests/enemy_collision_test.py images/frame1.jpg。

黄色虚线：YOLO 检测框；绿色实线：Enemy 实际使用的碰撞箱；青色十字：寻路坐标。
默认输入已裁剪好的游戏窗口截图；完整桌面截图可加 --auto-crop。
"""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from zaomeng_hanghai.core.constants import MAZE_HEIGHT, MAZE_WIDTH
from zaomeng_hanghai.core.entities import EnemyList
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.game_window import GameWindow, GameWindowError


PROJECT_DIR = Path(__file__).resolve().parent.parent


def draw_enemies(image, window, enemies, title):
    """将逻辑地图中的整数碰撞坐标映射回输入图片，保持与 Env.render 一致。"""
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    for index, box in enumerate(window.enemy_boxes or [], 1):
        x1, y1, x2, y2 = box.xyxyn
        ax.add_patch(Rectangle(
            (x1 * width, y1 * height), (x2 - x1) * width, (y2 - y1) * height,
            fill=False, edgecolor="#ffd54f", linestyle="--", linewidth=1.4,
        ))
        ax.text(x1 * width, y1 * height, f"E{index} {box.conf:.2f}", color="white", fontsize=9,
                verticalalignment="bottom", bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
    for enemy in enemies:
        left, right = enemy.x1 * width / MAZE_WIDTH, enemy.x2 * width / MAZE_WIDTH
        top, bottom = enemy.y1 * height / MAZE_HEIGHT, enemy.y2 * height / MAZE_HEIGHT
        ax.add_patch(Rectangle(
            (left, top), right - left, bottom - top,
            edgecolor="#00e676", facecolor=(0, 0.9, 0.3, 0.18), linewidth=2,
        ))
        ax.plot(enemy.x * width / MAZE_WIDTH, enemy.y * height / MAZE_HEIGHT,
                marker="+", color="cyan", markersize=10, markeredgewidth=1.5)
    ax.legend(handles=[
        Line2D([], [], color="#ffd54f", linestyle="--", label="YOLO detection"),
        Patch(edgecolor="#00e676", facecolor=(0, 0.9, 0.3, 0.18), label="Enemy collision box"),
        Line2D([], [], color="cyan", marker="+", linestyle="none", label="Navigation position"),
    ], loc="upper left", fontsize=9)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="游戏窗口 RGB 截图，路径相对当前工作目录")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "weights/best.onnx")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--map-index", type=int, choices=range(7), default=0, help="大地图编号，默认 0")
    parser.add_argument("--level", type=int, choices=range(1, 5), default=1, help="关卡编号，默认 1")
    parser.add_argument("--auto-crop", action="store_true", help="使用 GameWindow 从完整截图定位、裁剪游戏窗口")
    parser.add_argument("--force-detect", action="store_true", help="忽略加载/结算/血量判定，仍执行检测")
    parser.add_argument("--save", type=Path, help="保存可视化结果")
    parser.add_argument("--no-show", action="store_true", help="不弹出图片窗口")
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1:
        parser.error("--conf 必须在 0～1 之间")
    for path in (args.image, args.model):
        if not path.is_file():
            parser.error(f"文件不存在：{path}")
    try:
        with Image.open(args.image) as source:
            image = np.array(ImageOps.exif_transpose(source).convert("RGB"))
    except (OSError, UnidentifiedImageError) as error:
        parser.error(f"无法读取图片：{error}")

    window = GameWindow(str(args.model), capture_enabled=False)
    if args.auto_crop:
        try:
            x1, y1, x2, y2 = window.locate_window(cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
        except GameWindowError as error:
            parser.error(f"无法定位游戏窗口：{error}；请先手动裁剪图片")
        print(f"Crop: {(x1, y1, x2, y2)}")
        image = image[y1:y2, x1:x2].copy()
    height, width = image.shape[:2]
    window.window_size = (width, height)
    window.window_box = (0, 0, width, height)
    start = perf_counter()
    status = window.update_from_image(image, conf_threshold=args.conf, check_status=not args.force_detect)
    elapsed_ms = (perf_counter() - start) * 1000
    print(f"GameWindow status={status}, HP={window.hit_point}, perception={elapsed_ms:.2f} ms")
    if status:
        print("未进入实体检测流程；检查截图裁剪，或使用 --force-detect 忽略状态检查。")

    game_map = Map(render_enabled=False)
    game_map.reset(args.map_index)
    for _ in range(args.level):
        game_map.next_level()
    enemies = EnemyList()
    enemies.update(window.enemy_boxes or [], game_map.get_maze())
    print(f"Map={args.map_index}, level={args.level}, task={game_map.get_task()}, enemies={len(enemies)}")
    for index, enemy in enumerate(enemies, 1):
        print(f"E{index}: collision=({enemy.x1}, {enemy.y1}, {enemy.x2}, {enemy.y2}) "
              f"navigation=({enemy.x}, {enemy.y}) [320x200 coordinates]")
    fig = draw_enemies(image, window, enemies,
                       f"{args.image.name} | {len(enemies)} enemies | status={status} | {elapsed_ms:.1f} ms")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
