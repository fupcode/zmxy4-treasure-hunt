"""离线复现截图定位失败，导出格网和七张地图评分，不操作游戏。"""

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zmxy4_treasure_hunt.maze.diagnostics import export_alignment_debug
from zmxy4_treasure_hunt.maze.map_templates import load_weekday_map
from zmxy4_treasure_hunt.maze.mask import MaskRestorer
from zmxy4_treasure_hunt.maze.navigation import Maze
from zmxy4_treasure_hunt.maze.vision import detect_walls, locate_character, locate_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--cropped", action="store_true", help="输入是已裁剪的迷宫视区，例如失败现场的 raw.png")
    parser.add_argument("--weekday", type=int, choices=range(7), help="离线选择地图，星期日为 0；默认北京时间")
    parser.add_argument("--state", type=Path, help="读取失败现场 report.json，复现跟踪位置、移动历史和格距")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"无法读取图片：{args.image}")
    if args.cropped:
        raw = image
        region = (0, 0, image.shape[1], image.shape[0])
    else:
        (left, top), (right, bottom) = locate_game(image)
        raw = image[top:bottom, left:right]
        region = (left, top, right, bottom)
    saved = json.loads(args.state.read_text(encoding="utf-8")) if args.state else None
    weekday = args.weekday if args.weekday is not None else saved["weekday"] if saved else None
    template = load_weekday_map(weekday)
    maze = Maze(raw.shape[1], raw.shape[0], template.cells)
    if saved:
        state = saved["state"]
        maze.view_start = tuple(state["view_start"]) if state["view_start"] is not None else None
        maze.char_coord = tuple(state["char_coord"])
        maze.move_history.extend(state["move_history"])
        for name, value in state["grid"].items():
            setattr(maze.grid, name, value)
        for cell in state["removed_treasures"]:
            maze.remove_treasure(tuple(cell))
    frame = character = observation = None
    error = None
    try:
        frame = MaskRestorer(raw.shape[1], raw.shape[0]).restore(raw)
        character = locate_character(frame.image, frame.valid_pixels)
        observation = detect_walls(frame.image, character, frame.valid_pixels)
        result = maze.analyze(observation, character)
        print(f"定位通过：人物 {maze.char_coord}，视图 {maze.view_start}，路径长度 {len(result.path)}")
    except ValueError as exc:
        error = exc
        print(f"定位失败：{exc}")
    output, report = export_alignment_debug(maze, template, error, raw=raw, frame=frame,
                                           character=character, observation=observation,
                                           region=region, output_dir=args.output_dir)
    print(f"当前地图：{template.weekday} {template.name}；诊断目录：{output}")
    for item in report.get("weekdays", []):
        fixed = item["fixed_start"]
        best = item["best_global"][0] if item["best_global"] else {}
        print(f"地图 {item['weekday']}：起点评分 {fixed.get('score', fixed.get('reason'))}；"
              f"全图最高 {best.get('score')}，位置 {best.get('start')}")
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
