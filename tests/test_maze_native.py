"""Rust 搜索与 Python 参考实现的路径语义对照。"""

import unittest
from unittest.mock import patch

import numpy as np

from zmxy4_treasure_hunt.voyage.core import maze_utils


@unittest.skipUnless(maze_utils._native_region_path is not None, 'build the PyO3 extension first')
class NativeMazeTests(unittest.TestCase):
    def assert_matches_python(self, maze, start, targets, distance=None, **kwargs):
        before = [a.copy() for a in (maze, targets)]
        with patch.object(maze_utils, '_native_region_path', None):
            expected = maze_utils._region_path(maze, start, targets, distance, 5, **kwargs)
        actual = maze_utils._region_path(maze, start, targets, distance, 5, **kwargs)
        self.assertEqual(actual, expected)
        for original, snapshot in zip((maze, targets), before):
            np.testing.assert_array_equal(original, snapshot)

    def test_random_maps_match_python_paths(self):
        rng = np.random.default_rng(20261027)
        for case in range(100):
            maze = rng.random((18, 23)) < 0.22
            start = (int(rng.integers(23)), int(rng.integers(18)))
            maze[start[1], start[0]] = False
            targets = np.zeros(maze.shape, dtype=np.float32)
            targets[rng.random(maze.shape) < 0.03] = 0.7 if case % 4 == 0 else 1
            kwargs = dict(stride=(1, 5)[case % 2], allow_partial=case % 3 != 0,
                          last_direction=((0, 1), (1, 0), (0, -1), (-1, 0),
                                          (1, 1), (1, -1), (-1, 1), (-1, -1))[case % 8],
                          turn_cost=(0.0, 1.0, 2.0)[case % 3])
            if case % 3 == 0:
                kwargs['danger'] = rng.integers(0, 5, size=maze.shape)
            if case % 3 == 1:
                kwargs['cost_map'] = rng.random(maze.shape) * 4
            with self.subTest(case=case):
                self.assert_matches_python(maze, start, targets, **kwargs)

    def test_low_score_selection_searches_all_reachable_candidates(self):
        maze = np.zeros((16, 20), dtype=bool)
        targets = np.zeros(maze.shape)
        targets[2, 3] = 0.2
        targets[10, 15] = 0.7
        self.assert_matches_python(maze, (2, 2), targets)
        path = maze_utils._region_path(maze, (2, 2), targets, None, 5)
        self.assertEqual(path[-1], (15, 10))

    def test_contiguous_readonly_maps_are_passed_without_copying(self):
        maze = np.zeros((16, 20), dtype=bool)
        targets = np.zeros(maze.shape, dtype=np.float64)
        targets[10, 15] = 1
        distance = np.ones(maze.shape, dtype=np.float64)
        danger = np.zeros(maze.shape, dtype=np.float64)
        extra = np.ones(maze.shape, dtype=np.float64)
        for array in (maze, targets, distance, danger, extra):
            array.flags.writeable = False
        with patch.object(maze_utils, '_native_region_path', wraps=maze_utils._native_region_path) as search:
            path = maze_utils._region_path(
                maze, (2, 2), targets, distance, 5, danger=danger, cost_map=extra)
        self.assertEqual(path[-1], (15, 10))
        args = search.call_args.args
        for received, original in zip(
                (args[0], args[2], args[4], args[7], args[11]),
                (maze, targets, distance, danger, extra)):
            self.assertIs(received, original)

    def test_noncontiguous_maps_and_integer_inputs(self):
        maze = np.zeros((32, 40), dtype=np.uint8)[::2, ::2]
        maze[:12, 10] = 1
        targets = np.zeros((32, 40), dtype=np.float32)[::2, ::2]
        targets[2, 15] = 1
        distance = np.full((32, 40), 4, dtype=np.int32)[::2, ::2]
        extra = np.ones((32, 40), dtype=np.int64)[::2, ::2]
        self.assert_matches_python(maze, (2, 2), targets, distance, cost_map=extra)

    def test_unreachable_safe_area_preserves_partial_escape(self):
        maze = np.zeros((10, 10), dtype=bool)
        maze[:, 5] = True
        danger = np.tile(np.arange(10, 0, -1), (10, 1))
        danger[:, 8:] = 0
        targets = (danger == 0).astype(float)
        for partial in (True, False):
            with self.subTest(partial=partial):
                self.assert_matches_python(maze, (1, 5), targets, danger=danger,
                                           allow_partial=partial)

    def test_native_rejects_invalid_shapes_and_stride(self):
        maze = np.zeros((5, 8), dtype=bool)
        targets = np.ones(maze.shape)
        with self.assertRaisesRegex(ValueError, 'same shape'):
            maze_utils._region_path(maze, (1, 1), targets, np.ones((8, 5)), 1)
        with self.assertRaisesRegex(ValueError, 'stride must be positive'):
            maze_utils._region_path(maze, (1, 1), targets, None, 1, stride=0)


if __name__ == '__main__':
    unittest.main()
