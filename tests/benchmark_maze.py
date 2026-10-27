"""运行：python tests/benchmark_maze.py；比较包含预处理和数据转换的完整寻路耗时。"""

import statistics
import time
from unittest.mock import patch

import numpy as np

from zmxy4_treasure_hunt.voyage.core import maze_utils


def measure(search):
    search()
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        path = search()
        timings.append((time.perf_counter() - started) * 1000)
    return statistics.median(timings), path


def main():
    if maze_utils._native_region_path is None:
        raise SystemExit('Build the extension with maturin develop --release first.')
    maze = np.zeros((200, 320), dtype=bool)
    maze[[0, -1], :] = True
    maze[:, [0, -1]] = True
    blocked = maze.copy()
    blocked[:, 160] = True
    for name, grid, goal in (
        ('coarse reachable', maze, (282, 152)),
        ('off-grid reachable', maze, (282, 153)),
        ('unreachable', blocked, (282, 152)),
    ):
        targets = np.zeros(grid.shape)
        targets[goal[1], goal[0]] = 1
        distance = maze_utils.calculate_distance_from_obstacles(grid)

        def search():
            return maze_utils.find_best_path_to_region(
                grid, (22, 22), targets, distance_maze=distance, hate_factor=5)

        with patch.object(maze_utils, '_native_region_path', None):
            python_ms, expected = measure(search)
        rust_ms, actual = measure(search)
        if actual != expected:
            raise AssertionError(f'{name}: native path differs from Python')
        print(f'{name}: Python {python_ms:.3f} ms, Rust {rust_ms:.3f} ms, '
              f'{python_ms / rust_ms:.1f}x')


if __name__ == '__main__':
    main()
