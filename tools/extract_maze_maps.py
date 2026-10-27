"""解析 images/maze/ft0.png 至 ft6.png 的完整地图，不操作游戏。

格网位置和宝箱模板来自这批 1210×1098 图片；不用于任意游戏截图。
在项目根目录运行：python tools/extract_maze_maps.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = {"#": 2, ".": 1, "+": 3, "T": 4}
WEEKDAYS = ("星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六")
X = np.linspace(5, 1204, 29)
Y = np.linspace(5, 1092, 29)


def read_image(path):
    image = cv2.imread(str(path))
    if image is None or image.shape != (1098, 1210, 3):
        raise ValueError(f"图片尺寸应为 1210×1098：{path}")
    return image


def extract_grid(image, template):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grid = np.full((29, 29), "#", dtype="<U1")
    grid[1::2, 1::2] = "."
    # 只判断相邻房间之间的连接格。石墙低饱和，沙路高饱和。
    # 交点遵循现有导航的墙格约定，避免将墙头的草苔识别成通路。
    confidence = []
    for row in range(1, 28):
        for col in range(1, 28):
            if (row + col) % 2 == 0:
                continue
            x, y = round(X[col]), round(Y[row])
            saturation = float(np.median(hsv[y - 3:y + 4, x - 3:x + 4, 1]))
            # 这批图中墙壁中位饱和度 <=34，通路 >=73，55 位于二者之间。
            if 45 <= saturation <= 65:
                raise ValueError(f"连接格分类不确定：({row}, {col})，饱和度 {saturation}")
            grid[row, col] = "#" if saturation < 55 else "+"
            confidence.append(abs(saturation - 55))

    response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    treasures = []
    scores = []
    while True:
        _, score, _, (x, y) = cv2.minMaxLoc(response)
        if score < 0.9:
            break
        center_x = x + (template.shape[1] - 1) / 2
        center_y = y + (template.shape[0] - 1) / 2
        row = int(np.argmin(abs(Y - center_y)))
        col = int(np.argmin(abs(X - center_x)))
        if row % 2 != 1 or col % 2 != 1 or grid[row, col] != ".":
            raise ValueError(f"宝箱没有落在独立房间格：({row}, {col})")
        grid[row, col] = "T"
        treasures.append([row, col])
        scores.append(score)
        # 同一精灵的相邻高分响应只保留一次。
        response[max(0, y - 25):y + 26, max(0, x - 25):x + 26] = -1
    if len(treasures) != 8:
        raise ValueError(f"应找到 8 个宝箱，实际为 {len(treasures)}")
    return grid, sorted(treasures), min(confidence), min(scores)


def validate_connectivity(grid):
    seen = {(1, 1)}
    pending = [(1, 1)]
    while pending:
        row, col = pending.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = row + 2 * dy, col + 2 * dx
            if (0 < neighbor[0] < 28 and 0 < neighbor[1] < 28
                    and grid[row + dy, col + dx] == "+" and neighbor not in seen):
                seen.add(neighbor)
                pending.append(neighbor)
    if len(seen) != 196:
        raise ValueError(f"房间未全部连通：{len(seen)}/196")


def save_overlay(image, grid, output):
    """在源图上标出连接格判断和宝箱坐标，供人工复核。"""
    canvas = image.copy()
    for row in range(1, 28):
        for col in range(1, 28):
            x, y = round(X[col]), round(Y[row])
            if grid[row, col] == "T":
                cv2.rectangle(canvas, (x - 29, y - 29), (x + 29, y + 29), (0, 0, 255), 2)
                cv2.putText(canvas, f"{row},{col}", (x - 26, y - 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            elif (row + col) % 2:
                color = (255, 0, 255) if grid[row, col] == "#" else (255, 0, 0)
                cv2.circle(canvas, (x, y), 4, color, -1)
    cv2.imwrite(str(output), canvas)


def main():
    inputs = ROOT / "images" / "maze"
    output = ROOT / "maps"
    previews = inputs / "parsed"
    output.mkdir(exist_ok=True)
    previews.mkdir(exist_ok=True)
    # ft0 中的完整宝箱内部纹理；其余图片复用同一精灵。
    template = read_image(inputs / "ft0.png")[256:301, 371:413].copy()
    for weekday, name in enumerate(WEEKDAYS):
        image = read_image(inputs / f"ft{weekday}.png")
        grid, treasures, margin, score = extract_grid(image, template)
        validate_connectivity(grid)
        data = {
            "schema_version": 1,
            "weekday": weekday,
            "name": name,
            "source_image": f"images/maze/ft{weekday}.png",
            "size": [29, 29],
            "legend": SYMBOLS,
            "grid": ["".join(row) for row in grid],
            "treasures": treasures,
        }
        (output / f"{weekday}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        save_overlay(image, grid, previews / f"ft{weekday}.png")
        print(f"{weekday}: 29×29，196 个房间，8 个宝箱；连接格阈值最小余量 {margin:g}，宝箱最低匹配 {score:.6f}")


if __name__ == "__main__":
    main()
