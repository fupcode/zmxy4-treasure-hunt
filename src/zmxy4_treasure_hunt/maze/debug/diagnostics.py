"""导出失败画面与格网，离线比较星期地图；不修改运行时的定位决定。"""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from ..navigation import Maze, load_weekday_map


def _save_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"无法编码诊断图片：{path}")
    path.write_bytes(encoded.tobytes())


def _candidate(maze, view, view_char, start):
    row, col = start
    h, w = view.shape
    rows, cols = maze.layout.shape
    result = {"start": [row, col]}
    if not (0 <= view_char[0] < h and 0 <= view_char[1] < w):
        return {**result, "reason": "invalid_player_cell"}
    if row < 0 or col < 0 or row + h > rows or col + w > cols:
        return {**result, "reason": "out_of_bounds"}
    if maze.layout[row + view_char[0], col + view_char[1]] == 2:
        return {**result, "reason": "player_in_wall"}
    return {**result, **maze.comparison(view, start)}


def rank_weekdays(view, view_char, parity):
    """固定起点与全图搜索仅用于诊断，不让程序自动切图或跳转位置。"""
    rankings = []
    for weekday in range(7):
        try:
            template = load_weekday_map(weekday)
        except (OSError, ValueError, KeyError) as exc:
            rankings.append({"weekday": weekday, "error": str(exc),
                             "fixed_start": {"reason": "invalid_map"}, "best_global": []})
            continue
        maze = Maze(1, 1, template.cells)
        fixed = _candidate(maze, view, view_char, (1 - view_char[0], 1 - view_char[1]))
        candidates = []
        for row in range(parity[0], maze.layout.shape[0] - view.shape[0] + 1, 2):
            for col in range(parity[1], maze.layout.shape[1] - view.shape[1] + 1, 2):
                result = _candidate(maze, view, view_char, (row, col))
                if "score" in result:
                    candidates.append(result)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        rankings.append({"weekday": weekday, "name": template.name,
                         "fixed_start": fixed, "best_global": candidates[:3]})
    return rankings


def _grid_image(cells):
    colors = np.array([(90, 90, 90), (235, 235, 235), (0, 0, 0),
                       (235, 235, 235), (0, 190, 255), (60, 210, 60)], np.uint8)
    return np.repeat(np.repeat(colors[cells], 20, axis=0), 20, axis=1)


def export_alignment_debug(maze, template, error, *, raw=None, frame=None, character=None,
                           observation=None, region=None, output_dir=None):
    output = Path(output_dir) if output_dir else Path("runs/maze-debug") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=True)
    grid = maze.grid
    grid_state = {name: getattr(grid, name) for name in (
        "vertical_lines", "horizontal_lines", "vertical_start_small", "horizontal_start_small",
        "interval0x", "interval0y", "interval1x", "interval1y")}
    state = {"view_start": maze.view_start, "char_coord": maze.char_coord,
             "move_history": list(maze.move_history), "grid": grid_state,
             "removed_treasures": np.argwhere((maze.layout == 4) & (maze.maze != 4)).tolist()}
    report = {"error": str(error) if error else None, "weekday": template.weekday,
              "map_name": template.name, "region": region, "state": state,
              "character": asdict(character) if character else None,
              "alignment": maze.last_alignment}
    if raw is not None:
        _save_image(output / "raw.png", raw)
    if frame is not None:
        _save_image(output / "restored.png", frame.image)
        _save_image(output / "valid_pixels.png", frame.valid_pixels.astype(np.uint8) * 255)
        overlay = frame.image.copy()
        for x in grid.vertical_lines or []:
            cv2.line(overlay, (int(x), 0), (int(x), overlay.shape[0] - 1), (0, 255, 0), 1)
        for y in grid.horizontal_lines or []:
            cv2.line(overlay, (0, int(y)), (overlay.shape[1] - 1, int(y)), (255, 100, 0), 1)
        if character:
            cv2.circle(overlay, (character.x, character.y), character.radius, (0, 0, 255), 2)
            if character.ring_radius is not None:
                cv2.circle(overlay, (character.x, character.y), round(character.ring_radius), (0, 255, 255), 1)
        _save_image(output / "grid_overlay.png", overlay)
    if observation is not None:
        _save_image(output / "walls.png", observation.image)
        report["detected_lines"] = {"x": observation.vertical_lines, "y": observation.horizontal_lines}
    if maze.last_view is not None:
        view = maze.last_view
        report["observed_grid"] = view.tolist()
        report["view_char"] = list(maze.last_view_char)
        parity = (1 - int(grid.horizontal_start_small), 1 - int(grid.vertical_start_small))
        report["weekdays"] = rank_weekdays(view, maze.last_view_char, parity)
        observed = _grid_image(view)
        _save_image(output / "observed.png", observed)
        candidates = (maze.last_alignment or {}).get("candidates", [])
        scored = [item for item in candidates if "score" in item]
        if scored:
            best = max(scored, key=lambda item: item["score"])
            row, col = best["start"]
            expected = maze.layout[row:row + view.shape[0], col:col + view.shape[1]]
            for r, c in best["mismatch_cells"] + best["unexpected_treasures"]:
                cv2.rectangle(observed, (c * 20, r * 20), (c * 20 + 19, r * 20 + 19), (0, 0, 255), 2)
            _save_image(output / "comparison.png", cv2.hconcat((observed, _grid_image(expected))))
    def encode(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=encode) + "\n",
                                        encoding="utf-8")
    return output.resolve(), report
