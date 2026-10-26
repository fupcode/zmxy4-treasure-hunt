"""手动碰撞可视化：python tests/collision_test.py images/boss.png --level 4。

左图：所有检测框、实体位置、碰撞范围、弹幕方向和掉落物。
右图：原始障碍框、膨胀后的寻路障碍、战斗禁区和出口。
默认输入完整游戏窗口截图，桌面截图可加 --auto-crop；不采集屏幕或发送按键。
"""

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from zaomeng_hanghai.core.constants import (
    BULLET_AVOID_HEIGHT, BULLET_AVOID_WIDTH,
    END_DOWN, END_LEFT, END_RIGHT, END_UP, EXIT_DOWN, EXIT_LEFT, EXIT_RIGHT, EXIT_UP,
    HIT_HEIGHT, HIT_WIDTH,
    MAP_DOWN, MAP_LEFT, MAP_RIGHT, MAP_UP, MAZE_HEIGHT, MAZE_WIDTH,
)
from zaomeng_hanghai.core.entities import Bullet, Bullets, EnemyList, FallingItems, Player
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.game_window import GameWindow, GameWindowError


PROJECT_DIR = Path(__file__).resolve().parent.parent
COLORS = {
    "detection": "#ffd54f", "enemy": "#00e676", "player": "#40c4ff",
    "bullet": "#ff4d73", "safe_bullet": "#b0bec5", "item": "#ff80ff",
    "obstacle": "#ff9800", "maze": "#b388ff", "forbidden": "#ff5252", "exit": "#ffeb3b",
}


def raw_obstacle_mask(game_map):
    """按 Map 的同一布局缩放规则还原未做碰撞膨胀的障碍。"""
    mask = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
    left, top = int(round(MAP_LEFT * MAZE_WIDTH)), int(round(MAP_UP * MAZE_HEIGHT))
    width = int(round((MAP_RIGHT - MAP_LEFT) * MAZE_WIDTH))
    height = int(round((MAP_DOWN - MAP_UP) * MAZE_HEIGHT))
    mask[top:top + height, left:left + width] = game_map.enlarge_bool_array(
        game_map.task_maps[game_map.get_task()], width, height
    )
    return mask


def mask_rectangles(mask):
    """合并逐行连续区间，矩形集合精确覆盖障碍，不把凹形空地圈成障碍。"""
    active, rectangles = {}, []
    for y in range(mask.shape[0] + 1):
        row = mask[y] if y < mask.shape[0] else np.zeros(mask.shape[1], dtype=bool)
        changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        runs = set(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        for interval in list(active):
            if interval not in runs:
                rectangles.append((int(interval[0]), active.pop(interval), int(interval[1]), y))
        for interval in sorted(runs):
            active.setdefault(interval, y)
    return rectangles


def add_box(ax, bounds, image_size, color, *, label=None, fill_alpha=0, linestyle="-", linewidth=1.3):
    """bounds 使用 320×200 逻辑坐标；右下为几何边界，允许等于宽/高。"""
    width, height = image_size
    x1, y1, x2, y2 = bounds
    x1, x2 = np.clip([x1, x2], 0, MAZE_WIDTH) * width / MAZE_WIDTH
    y1, y2 = np.clip([y1, y2], 0, MAZE_HEIGHT) * height / MAZE_HEIGHT
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1, edgecolor=color,
        facecolor=(*to_rgb(color), fill_alpha),
        linestyle=linestyle, linewidth=linewidth,
    ))
    if label:
        ax.text(x1 + 2, y1 + 2, label, color=color, fontsize=8, va="top",
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 1, "edgecolor": "none"})


def add_mask(ax, mask, image_size, color, alpha):
    if not mask.any():
        return
    width, height = image_size
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask] = (*to_rgb(color), alpha)
    ax.imshow(rgba, extent=(0, width, height, 0), interpolation="nearest")
    # 把网格看作单位面积，轮廓落在单元边界而非像素中心。
    xs = (np.arange(mask.shape[1] + 2) - 0.5) * width / MAZE_WIDTH
    ys = (np.arange(mask.shape[0] + 2) - 0.5) * height / MAZE_HEIGHT
    ax.contour(xs, ys, np.pad(mask, 1).astype(float), levels=[0.5], colors=[color], linewidths=0.8)


def add_position(ax, entity, name, image_size, color):
    width, height = image_size
    raw_x, raw_y = entity.xn * width, entity.yn * height
    x, y = entity.x * width / MAZE_WIDTH, entity.y * height / MAZE_HEIGHT
    ax.plot(raw_x, raw_y, "x", color="white", markersize=5, markeredgewidth=1)
    ax.plot([raw_x, x], [raw_y, y], color="white", linewidth=0.7)
    ax.plot(x, y, "+", color=color, markersize=9, markeredgewidth=1.6)
    ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", color=color, fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 1, "edgecolor": "none"})


def add_bullet(ax, bullet, name, image_size, *, safe=False, show_lane=False):
    color = COLORS["safe_bullet" if safe else "bullet"]
    width, height = image_size
    dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[bullet.d]
    add_position(ax, bullet, name, image_size, color)
    ax.annotate("", xy=((bullet.x + dx * 12) * width / MAZE_WIDTH,
                         (bullet.y + dy * 12) * height / MAZE_HEIGHT),
                xytext=(bullet.x * width / MAZE_WIDTH, bullet.y * height / MAZE_HEIGHT),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.6})
    add_box(ax, (bullet.x - BULLET_AVOID_WIDTH, bullet.y - BULLET_AVOID_HEIGHT,
                 bullet.x + BULLET_AVOID_WIDTH + 1, bullet.y + BULLET_AVOID_HEIGHT + 1),
            image_size, color, linestyle=":")
    if show_lane and not safe:
        bounds = Map.get_bullet_danger_bounds(bullet)
        add_box(ax, bounds, image_size, color, fill_alpha=0.08, linewidth=0.7)


def draw_scene(image, window, game_map, player, enemies, bullets, items, title):
    height, width = image.shape[:2]
    size = (width, height)
    fig, (entities_ax, map_ax) = plt.subplots(1, 2, figsize=(22, 9))
    for ax in (entities_ax, map_ax):
        ax.imshow(image)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
    entities_ax.set_title("Detections, entity boxes and positions", fontsize=12)
    map_ax.set_title("Map obstacles, movement collision, exits and bullet lanes", fontsize=12)
    for kind, detections in window.detections.items():
        for index, box in enumerate(detections, 1):
            x1, y1, x2, y2 = box.xyxyn
            add_box(entities_ax, (x1 * MAZE_WIDTH, y1 * MAZE_HEIGHT, x2 * MAZE_WIDTH, y2 * MAZE_HEIGHT),
                    size, COLORS["detection"], label=f"{kind}{index} {box.conf:.2f}", linestyle="--", linewidth=1)

    maze = game_map.get_maze()
    obstacle_mask = raw_obstacle_mask(game_map)
    add_mask(map_ax, maze, size, COLORS["maze"], 0.2)
    extra_forbidden = game_map.map_template_without_exit & ~maze
    add_mask(map_ax, extra_forbidden, size, COLORS["forbidden"], 0.24)
    for index, bounds in enumerate(mask_rectangles(obstacle_mask), 1):
        add_box(map_ax, bounds, size, COLORS["obstacle"], label=f"O{index}", fill_alpha=0.15, linewidth=1.8)
    add_box(map_ax, (MAP_LEFT * MAZE_WIDTH, MAP_UP * MAZE_HEIGHT,
                     MAP_RIGHT * MAZE_WIDTH, MAP_DOWN * MAZE_HEIGHT), size, "white", linestyle="--", linewidth=0.8)
    exit_regions = [
        (MAP_RIGHT, EXIT_UP, 1, EXIT_DOWN), (EXIT_LEFT, MAP_DOWN, EXIT_RIGHT, 1),
        (0, EXIT_UP, MAP_LEFT, EXIT_DOWN), (EXIT_LEFT, 0, EXIT_RIGHT, MAP_UP),
        (END_LEFT, END_UP, END_RIGHT, END_DOWN),
    ]
    for index, (x1, y1, x2, y2) in enumerate(exit_regions):
        if index == 4 and game_map.level != 4:
            continue
        selected = index == game_map.next_exit
        add_box(map_ax, (x1 * MAZE_WIDTH, y1 * MAZE_HEIGHT, x2 * MAZE_WIDTH, y2 * MAZE_HEIGHT), size,
                COLORS["exit"] if selected else "#b0bec5", label="GOAL" if index == 4 else f"EXIT {index}",
                linestyle="-" if selected else ":", linewidth=2 if selected else 0.8)

    for ax in (entities_ax, map_ax):
        for index, enemy in enumerate(enemies, 1):
            add_box(ax, (enemy.x1, enemy.y1, enemy.x2, enemy.y2), size, COLORS["enemy"], fill_alpha=0.12)
            add_position(ax, enemy, f"E{index}", size, COLORS["enemy"])
        if player is not None:
            add_position(ax, player, "P", size, COLORS["player"])
            # HIT 常量是避弹判定范围，Player 类并未定义独立的真实碰撞框。
            add_box(ax, (player.x - HIT_WIDTH, player.y - HIT_HEIGHT,
                         player.x + HIT_WIDTH, player.y + HIT_HEIGHT), size, COLORS["player"], linestyle=":")
        for index, bullet in enumerate(bullets, 1):
            add_bullet(ax, bullet, f"B{index}", size, show_lane=ax is map_ax and player is not None)
        for index, bullet in enumerate(bullets.safe_bullets, 1):
            add_bullet(ax, bullet, f"S{index}", size, safe=True)
        for index, item in enumerate(items, 1):
            add_position(ax, item, f"I{index}", size, COLORS["item"])
        if player is not None:
            direction = game_map.fire_direction(player, enemies)
            if direction is not None:
                for enemy in enemies:
                    distance, row = game_map.fire_distance(player, enemy)
                    if distance is not None:
                        target = (enemy.x, player.y) if row else (player.x, enemy.y)
                        ax.plot([player.x * width / MAZE_WIDTH, target[0] * width / MAZE_WIDTH],
                                [player.y * height / MAZE_HEIGHT, target[1] * height / MAZE_HEIGHT],
                                color=COLORS["enemy"], linestyle="--", linewidth=1)

    handles = [
        Line2D([], [], color=COLORS["detection"], linestyle="--", label="All YOLO detection boxes"),
        Patch(edgecolor=COLORS["enemy"], facecolor="none", label="Enemy controller box (E)"),
        Line2D([], [], color=COLORS["player"], linestyle=":", label="Player HIT envelope (P)"),
        Line2D([], [], color=COLORS["bullet"], marker=">", label="Danger bullet / lane (B)"),
        Line2D([], [], color=COLORS["safe_bullet"], marker=">", label="Passed bullet (S)"),
        Line2D([], [], color=COLORS["item"], marker="+", linestyle="none", label="Useful drop (I)"),
        Patch(edgecolor=COLORS["obstacle"], facecolor="none", label="Raw map obstacle (O)"),
        Patch(color=COLORS["maze"], alpha=0.4, label="Expanded navigation obstacle"),
        Patch(color=COLORS["forbidden"], alpha=0.4, label="Extra combat forbidden area"),
        Line2D([], [], color=COLORS["exit"], label="Current exit / goal"),
        Line2D([], [], color="white", marker="x", linestyle="none", markeredgecolor="black", label="Raw entity position (x)"),
        Line2D([], [], color="black", marker="+", linestyle="none", label="Rounded / snapped position (+)"),
    ]
    fig.suptitle(title, fontsize=13)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, framealpha=0.95)
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.01, right=0.99, wspace=0.035)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="游戏窗口截图，路径相对当前工作目录")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "weights/best.onnx")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--map-index", type=int, choices=range(7), default=0, help="大地图编号，默认 0")
    parser.add_argument("--level", type=int, choices=range(1, 5), default=1, help="关卡编号，默认 1")
    parser.add_argument("--auto-crop", action="store_true", help="从桌面截图定位并裁剪游戏窗口")
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
    if width < 100 or height < 100:
        parser.error("请提供完整游戏窗口截图；局部物体小图无法使用 GameWindow 的界面解析。")
    window.window_size, window.window_box = (width, height), (0, 0, width, height)
    start = perf_counter()
    status = window.update_from_image(image, conf_threshold=args.conf, check_status=not args.force_detect)
    elapsed_ms = (perf_counter() - start) * 1000
    print(f"GameWindow status={status}, HP={window.hit_point}, perception={elapsed_ms:.2f} ms")
    if status:
        print("未进入实体检测流程；检查截图裁剪，或加 --force-detect。地图叠加仍按指定关卡显示。")
    game_map = Map(render_enabled=False)
    game_map.reset(args.map_index)
    for _ in range(args.level):
        game_map.next_level()
    maze = game_map.get_maze()
    player = Player(window.player_box, maze) if window.player_box is not None else None
    enemies, bullets, items = EnemyList(), Bullets(), FallingItems()
    enemies.update(window.enemy_boxes or [], maze)
    items.update(window.falling_items_normal or [])
    if player is not None:
        bullets.update(window.bullet_list_normal or [], player)
    else:
        bullets.data = [Bullet(info) for info in (window.bullet_list_normal or [])]
        print("没有玩家检测结果，弹幕未做安全/危险分类。")
    for kind, boxes in window.detections.items():
        print(f"YOLO {kind}: {len(boxes)}")
    print(f"Map={args.map_index}, level={args.level}, task={game_map.get_task()}, "
          f"obstacle rectangles={len(mask_rectangles(raw_obstacle_mask(game_map)))}")
    print(f"Player={player is not None}, enemies={len(enemies)}, danger bullets={len(bullets)}, "
          f"passed bullets={len(bullets.safe_bullets)}, useful drops={len(items)}, items={window.items}")
    if player is not None:
        print(f"P: normalized=({player.xn:.6f}, {player.yn:.6f})")
        print("Relative normalized coordinates: enemy - player; +dx=right, +dy=down")
    for index, enemy in enumerate(enemies, 1):
        print(f"E{index}: box=({enemy.x1}, {enemy.y1}, {enemy.x2}, {enemy.y2}) "
              f"raw=({enemy.xn * MAZE_WIDTH:.2f}, {enemy.yn * MAZE_HEIGHT:.2f}) "
              f"navigation=({enemy.x}, {enemy.y}) "
              f"normalized=({enemy.xn:.6f}, {enemy.yn:.6f})")
        if player is not None:
            print(f"  relative_to_player=(dx={enemy.xn - player.xn:+.6f}, "
                  f"dy={enemy.yn - player.yn:+.6f})")
    title = (f"{args.image.name} | map={args.map_index} level={args.level} task={game_map.get_task()} "
             f"| status={status} HP={window.hit_point} | {elapsed_ms:.1f} ms\n"
             "Static layout comes from the selected map/level; single-image cooldown is unknown.")
    fig = draw_scene(image, window, game_map, player, enemies, bullets, items, title)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
