"""迷宫拼接、探索路径和移动建议；不依赖屏幕或键盘。"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .grid import Grid

UNKNOWN, ROAD, WALL, PASSAGE, TREASURE, PLAYER = range(6)


class MapAlignmentError(ValueError):
    """当前局部视图无法与累计地图对齐。"""


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
                    if (self._inside(target) and maze[row + dy, col + dx] in (PASSAGE, UNKNOWN)
                            and maze[target] != WALL):
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
        return [target for target in candidates if self._inside(target) and self.maze[target] != WALL]

    def bfs(self, char_coord, greedy=2):
        """保留探索策略：宝箱路径小于最近未知路径的 greedy 倍时优先宝箱。"""
        start = tuple(char_coord)
        if not self._inside(start) or self.maze[start] == WALL:
            return []
        parent = {start: None}
        queue = deque([start])
        unknown_path = []
        while queue:
            node = queue.popleft()
            if self.maze[node] == UNKNOWN and not unknown_path:
                unknown_path = self.get_path(node, parent)
            elif self.maze[node] == TREASURE:
                treasure_path = self.get_path(node, parent)
                if not unknown_path or len(treasure_path) < len(unknown_path) * greedy:
                    return treasure_path
                return unknown_path
            for neighbor in self._neighbors(node):
                if neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return unknown_path

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


class Maze:
    def __init__(self, width, height, size=29):
        self.grid = Grid(width, height)
        self.size = size
        self.maze = None
        self.graph = None
        self.view_start = (0, 0)
        self.char_coord = (1, 1)
        self.char_offset = (0, 0)
        self.move_history = deque(maxlen=3)

    def compare(self, view, start):
        h, w = view.shape
        known = self.maze[start[0]:start[0] + h, start[1]:start[1] + w]
        mask = known != UNKNOWN
        return float(np.mean(view[mask] == known[mask])) if mask.any() else 0.0

    def slide(self, view):
        h, w = view.shape
        history = set(''.join(self.move_history))
        candidates = {}
        for dy in range(-2 if 'W' in history else 0, 3 if 'S' in history else 1):
            for dx in range(-2 if 'A' in history else 0, 3 if 'D' in history else 1):
                start = self.view_start[0] + dy, self.view_start[1] + dx
                if start[0] < 0 or start[1] < 0 or start[0] + h > self.size or start[1] + w > self.size:
                    continue
                score = self.compare(view, start)
                if score == 1:
                    return start
                if score > 0.88:
                    candidates[start] = score
        if candidates:
            return max(candidates, key=candidates.get)
        raise MapAlignmentError(f"无法对齐局部视图，当前位置 {self.view_start}，最近移动 {list(self.move_history)}")

    def record_movements(self, movements):
        """仅在控制器实际执行动作后记录，供下一帧拼图使用。"""
        self.move_history.append(movements)

    def remove_treasure(self, cell):
        """触发后消失的假宝箱恢复为道路，其余地图和定位状态保持不变。"""
        if self.maze is not None and self.maze[cell] == TREASURE:
            self.maze[cell] = ROAD
            self.graph = None

    def analyze(self, observation, character):
        self.grid.update(observation.vertical_lines, observation.horizontal_lines)
        view, view_char, self.char_offset = self.grid.analyze(
            observation.image, (character.x, character.y), observation.treasures
        )
        return self.load_view(view, view_char)

    def load_view(self, view, view_char):
        h, w = view.shape
        if h > self.size or w > self.size:
            raise MapAlignmentError("局部视图大于完整迷宫")
        if not (0 <= view_char[0] < h and 0 <= view_char[1] < w):
            raise MapAlignmentError("人物位置超出局部视图")
        if self.maze is None:
            self.view_start = (1 - int(self.grid.horizontal_start_small),
                               1 - int(self.grid.vertical_start_small))
            if self.view_start[0] + h > self.size or self.view_start[1] + w > self.size:
                raise MapAlignmentError("局部视图对齐后超出完整迷宫")
            self.maze = np.zeros((self.size, self.size), dtype=np.uint8)
            self.maze[0, :] = self.maze[-1, :] = WALL
            self.maze[:, 0] = self.maze[:, -1] = WALL
            self.maze[::2, ::2] = WALL
        else:
            self.view_start = self.slide(view)
        row, col = self.view_start
        self.maze[row:row + h, col:col + w] = view
        self.char_coord = row + int(view_char[0]), col + int(view_char[1])
        self.graph = Graph(self.maze)
        path = self.graph.bfs(self.char_coord)
        movements = self.generate_movements(path)
        marked = self.maze.copy()
        marked[self.char_coord] = PLAYER
        approaching = bool(path) and len(path) <= 2 and self.maze[path[-1]] == TREASURE
        return NavigationResult(marked, path, movements, approaching)

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
