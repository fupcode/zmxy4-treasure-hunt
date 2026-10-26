"""障碍距离变换与粗搜索跳过条件的回归测试。"""

import unittest
from unittest.mock import patch

import numpy as np

from zaomeng_hanghai.core import maze_utils


class MazeUtilsTests(unittest.TestCase):
    def test_distance_matches_nearest_obstacle_manhattan_distance(self):
        rng = np.random.default_rng(42)
        for shape in ((1, 7), (7, 1), (12, 17)):
            for dtype in (bool, np.uint8):
                with self.subTest(shape=shape, dtype=dtype):
                    maze = (rng.random(shape) < 0.25).astype(dtype)
                    maze[0, 0] = 1
                    before = maze.copy()
                    yy, xx = np.indices(shape)
                    expected = np.minimum.reduce([
                        yy + 1, xx + 1, shape[0] - yy, shape[1] - xx,
                    ] + [
                        abs(yy - y) + abs(xx - x)
                        for y, x in np.argwhere(maze == 1)
                    ])
                    actual = maze_utils.calculate_distance_from_obstacles(maze)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(maze, before)
                    self.assertEqual(actual.dtype, np.float64)

    def test_distance_without_internal_obstacles_measures_distance_to_outer_wall(self):
        for shape in ((5, 8), (1, 7), (7, 1), (1, 1)):
            with self.subTest(shape=shape):
                actual = maze_utils.calculate_distance_from_obstacles(np.zeros(shape, dtype=bool))
                yy, xx = np.indices(shape)
                expected = np.minimum.reduce([yy + 1, xx + 1, shape[0] - yy, shape[1] - xx])
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, shape)
                self.assertEqual(actual.dtype, np.float64)

    def test_distance_preserves_empty_array_shape(self):
        for shape in ((0, 8), (5, 0), (0, 0)):
            with self.subTest(shape=shape):
                actual = maze_utils.calculate_distance_from_obstacles(np.zeros(shape, dtype=bool))
                self.assertEqual(actual.shape, shape)
                self.assertEqual(actual.dtype, np.float64)

    def test_distance_with_only_obstacles_is_zero(self):
        actual = maze_utils.calculate_distance_from_obstacles(np.ones((5, 8), dtype=bool))
        np.testing.assert_array_equal(actual, np.zeros((5, 8)))

    def test_distance_supports_noncontiguous_input(self):
        maze = np.zeros((10, 16), dtype=bool)[::2, ::2]
        maze[2, 3] = True
        yy, xx = np.indices(maze.shape)
        expected = np.minimum.reduce([
            abs(yy - 2) + abs(xx - 3), yy + 1, xx + 1,
            maze.shape[0] - yy, maze.shape[1] - xx,
        ])
        np.testing.assert_array_equal(
            maze_utils.calculate_distance_from_obstacles(maze), expected)

    def test_coarse_search_skips_grid_without_passable_target(self):
        maze = np.zeros((25, 30), dtype=bool)
        maze[12, 22] = True  # 在粗网格上，但目标被墙阻挡。
        targets = np.zeros(maze.shape)
        targets[12, 22] = targets[13, 22] = 1
        before = targets.copy()
        with patch.object(maze_utils, '_region_path', wraps=maze_utils._region_path) as search:
            path = maze_utils.find_best_path_to_region(maze, (2, 2), targets)
        self.assertEqual(path[-1], (22, 13))
        self.assertEqual([call.kwargs['stride'] for call in search.call_args_list], [1])
        np.testing.assert_array_equal(targets, before)

    def test_turn_cost_increases_with_angle(self):
        costs = dict(zip(maze_utils._DIRECTIONS, maze_utils.calculate_turn_costs((1, 0), 2)))
        for direction, expected in (
            ((1, 0), 0), ((1, 1), 0.5), ((0, 1), 1), ((-1, 1), 1.5), ((-1, 0), 2),
        ):
            with self.subTest(direction=direction):
                self.assertEqual(costs[direction], expected)
        self.assertEqual(maze_utils.calculate_turn_costs(None, 2), (0,) * 8)
        for invalid in (-1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                maze_utils.calculate_turn_costs((1, 0), invalid)

    def test_region_turn_cost_prefers_same_direction_at_both_strides(self):
        maze = np.zeros((25, 30), dtype=bool)
        targets = np.zeros(maze.shape)
        targets[12, 2] = targets[12, 22] = 1
        for stride in (1, 5):
            with self.subTest(stride=stride):
                path = maze_utils._region_path(
                    maze, (12, 12), targets, None, 0, stride=stride,
                    last_direction=(1, 0), turn_cost=2)
                self.assertEqual(path[0], (12 + stride, 12))
                self.assertEqual(path[-1], (22, 12))

    def test_turn_cost_is_charged_only_on_first_step(self):
        maze = np.ones((5, 9), dtype=bool)
        maze[:4, 3] = False
        maze[3, 3:8] = False
        targets = np.zeros(maze.shape)
        targets[0, 3] = targets[3, 7] = 1
        # 向上 3 步 + 一次 90 度转向 0.5，比保持向右 4 步便宜。
        path = maze_utils._region_path(
            maze, (3, 3), targets, None, 0, last_direction=(1, 0), turn_cost=1)
        self.assertEqual(path[-1], (3, 0))

    def test_escape_can_reverse_when_only_safe_point_is_behind(self):
        maze = np.ones((3, 9), dtype=bool)
        maze[1, 1:8] = False
        danger = np.ones(maze.shape)
        danger[1, 1] = 0
        path = maze_utils.find_nearest_passable_path(
            maze, (6, 1), danger, hate_factor=0, last_direction=(1, 0))
        self.assertEqual(path[0], (1, 1))
        self.assertEqual(path[-1], (1, 1))

    def test_point_path_prefers_previous_direction(self):
        maze = np.zeros((20, 20), dtype=bool)
        path = maze_utils.find_path_to_point(
            maze, (2, 8), (12, 8), stride=1, hate_factor=0, last_direction=(1, 0))
        self.assertEqual(path[0], (3, 8))
        self.assertEqual(path[-1], (12, 8))


if __name__ == '__main__':
    unittest.main()
