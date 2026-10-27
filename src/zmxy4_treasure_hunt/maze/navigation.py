"""完整地图定位、宝箱最短路径和移动建议；不依赖屏幕或键盘。"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .grid import Grid

UNKNOWN, ROAD, WALL, PASSAGE, TREASURE, PLAYER = range(6)


class MapAlignmentError(ValueError):
    """当前局部视图无法在完整地图上可靠定位。"""


class Graph:
    def __init__(self, maze, r0=1, c0=1):
        self.maze = maze
        self.r0, self.c0 = r0, c0
        self.graph = {}
        rows, cols = maze.shape
        for row in range(r0, rows, 2):
            for col in range(c0, cols, 2):
                if maze[row, col] not in (ROAD, TREASURE):
                    continue
                neighbors = []
                for dy, dx in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    target = row + 2 * dy, col + 2 * dx
                    if (self._inside(target) and maze[row + dy, col + dx] == PASSAGE
                            and maze[target] in (ROAD, TREASURE)):
                        neighbors.append(target)
                self.graph[row, col] = neighbors

    def _inside(self, cell):
        return 0 <= cell[0] < self.maze.shape[0] and 0 <= cell[1] < self.maze.shape[1]

    def _neighbors(self, cell):
        row, col = cell
        if row % 2 == self.r0 and col % 2 == self.c0:
            return self.graph.get(cell, ())
        if row % 2 == self.r0:
            candidates = ((row, col - 1), (row, col + 1))
        elif col % 2 == self.c0:
            candidates = ((row - 1, col), (row + 1, col))
        else:
            return ()
        return [target for target in candidates if self._inside(target) and self.maze[target] in (ROAD, TREASURE)]

    def bfs(self, char_coord):
        """沿完整地图的通路寻找最近的剩余宝箱，不搜索未知格。"""
        start = tuple(char_coord)
        if not self._inside(start) or self.maze[start] == WALL:
            return []
        parent = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if self.maze[node] == TREASURE:
                return self.get_path(node, parent)
            for neighbor in self._neighbors(node):
                if neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return []

    @staticmethod
    def get_path(end, parent):
        path = []
        while end is not None:
            path.append(end)
            end = parent[end]
        return path[::-1]


@dataclass
class NavigationResult:
    image: np.ndarray
    path: list[tuple[int, int]]
    movements: str
    approaching_treasure: bool
    complete: bool = False


class Maze:
    def __init__(self, width, height, cells):
        cells = np.asarray(cells)
        if (cells.ndim != 2 or min(cells.shape) < 3 or
                not np.all(np.isin(cells, [ROAD, WALL, PASSAGE, TREASURE]))):
            raise ValueError("必须提供完整地图，不能含未知格")
        self.grid = Grid(width, height)
        self.layout = cells.astype(np.uint8, copy=True)
        self.layout.flags.writeable = False
        self.maze = self.layout.copy()
        self.graph = Graph(self.maze)
        self.view_start = None
        self.char_coord = (1, 1)
        self.char_offset = (0, 0)
        self.move_history = deque(maxlen=3)

    def _score(self, view, start):
        h, w = view.shape
        row, col = start
        expected = self.layout[row:row + h, col:col + w]
        # 房间和交点本来就具有固定奇偶排列，只比较能区分地图的连接格。
        rr, cc = np.indices(view.shape)
        mask = ((rr + row + cc + col) % 2 == 1) & (view != UNKNOWN)
        if not mask.any():
            return 0.0
        # 可见宝箱提供额外定位约束；看不到宝箱不表示模板中的宝箱已消失。
        if np.any((view == TREASURE) & (expected != TREASURE)):
            return 0.0
        return float(np.mean((view[mask] == WALL) == (expected[mask] == WALL)))

    def locate_view(self, view, view_char):
        h, w = view.shape
        rows, cols = self.maze.shape
        row_parity = 1 - int(self.grid.horizontal_start_small)
        col_parity = 1 - int(self.grid.vertical_start_small)
        if self.view_start is None:
            # 起点固定为 (1, 1)，用画面中的人物格反推首帧视图原点。
            candidates = [(self.char_coord[0] - int(view_char[0]),
                           self.char_coord[1] - int(view_char[1]))]
        else:
            history = set(''.join(self.move_history))
            candidates = ((self.view_start[0] + dy, self.view_start[1] + dx)
                          for dy in range(-2 if 'W' in history else 0, 3 if 'S' in history else 1)
                          for dx in range(-2 if 'A' in history else 0, 3 if 'D' in history else 1))
        scores = {}
        for row, col in candidates:
            if (row < 0 or col < 0 or row + h > rows or col + w > cols
                    or row % 2 != row_parity or col % 2 != col_parity):
                continue
            player = row + int(view_char[0]), col + int(view_char[1])
            if self.maze[player] == WALL:
                continue
            score = self._score(view, (row, col))
            if score >= 0.88:
                scores[row, col] = score
        if not scores:
            raise MapAlignmentError("画面与完整地图不匹配，请检查当日地图和格网识别")
        best_score = max(scores.values())
        best = [cell for cell, score in scores.items() if abs(score - best_score) < 1e-9]
        if len(best) == 1:
            return best[0]
        if self.view_start in best:
            return self.view_start
        raise MapAlignmentError("画面可匹配完整地图的多个位置，无法确定人物坐标")

    def record_movements(self, movements):
        """仅在控制器实际执行动作后记录，约束下一帧的定位范围。"""
        self.move_history.append(movements)

    def remove_treasure(self, cell):
        """触发后消失的假宝箱恢复为道路，其余地图和定位状态保持不变。"""
        if self.maze is not None and self.maze[cell] == TREASURE:
            self.maze[cell] = ROAD
            self.graph = None

    def analyze(self, observation, character):
        self.grid.update(observation.vertical_lines, observation.horizontal_lines)
        view, view_char, self.char_offset = self.grid.analyze(
            observation.image, (character.x, character.y), observation.treasures, observation.valid_pixels
        )
        return self.load_view(view, view_char)

    def load_view(self, view, view_char):
        view = np.asarray(view)
        h, w = view.shape
        if not (0 <= view_char[0] < h and 0 <= view_char[1] < w):
            raise MapAlignmentError("人物位置超出局部视图")
        start = self.locate_view(view, view_char)
        self.view_start = start
        self.char_coord = start[0] + int(view_char[0]), start[1] + int(view_char[1])
        if self.graph is None:
            self.graph = Graph(self.maze)
        path = self.graph.bfs(self.char_coord)
        complete = not np.any(self.maze == TREASURE)
        if not path and not complete:
            raise MapAlignmentError("当前位置无法到达剩余宝箱")
        movements = self.generate_movements(path)
        marked = self.maze.copy()
        marked[self.char_coord] = PLAYER
        approaching = bool(path) and len(path) <= 2 and self.maze[path[-1]] == TREASURE
        return NavigationResult(marked, path, movements, approaching, complete)

    def generate_movements(self, path, threshold=0.5):
        movements = ""
        if len(path) > 1:
            dy, dx = path[1][0] - path[0][0], path[1][1] - path[0][1]
            if dy == 0:  # 水平移动，竖直微调
                if self.char_offset[1] > threshold:
                    movements = "W"
                elif self.char_offset[1] < -threshold:
                    movements = "S"
                if dx > 0:
                    movements += "D"
                elif dx < 0:
                    movements += "A"
            elif dx == 0:  # 竖直移动，水平微调
                if self.char_offset[0] > threshold:
                    movements = "A"
                elif self.char_offset[0] < -threshold:
                    movements = "D"
                if dy > 0:
                    movements += "S"
                elif dy < 0:
                    movements += "W"

        return movements
