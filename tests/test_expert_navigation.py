"""专家战斗寻路的窄目标、墙角和敌人重叠回归测试。"""

import unittest
from types import SimpleNamespace

import numpy as np

from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.maze_utils import (
    find_best_path_to_region,
    find_nearest_passable_path,
)


class ExpertNavigationTests(unittest.TestCase):
    def test_single_pixel_target_off_stride_grid(self):
        maze = np.zeros((25, 30), dtype=bool)
        targets = np.zeros(maze.shape)
        targets[13, 22] = 0.9
        path = find_best_path_to_region(maze, (2, 2), targets,
                                        distance_maze=np.ones(maze.shape))
        self.assertEqual(path[-1], (22, 13))

    def test_thin_wall_and_single_pixel_door(self):
        maze = np.zeros((20, 30), dtype=bool)
        maze[:, 12] = True
        maze[7, 12] = False
        targets = np.zeros(maze.shape)
        targets[10, 22] = 0.9
        path = find_best_path_to_region(maze, (2, 10), targets,
                                        distance_maze=np.ones(maze.shape))
        self.assertEqual(path[-1], (22, 10))
        self.assertIn((12, 7), path)
        self.assertTrue(all(not maze[y, x] for x, y in path))

    def test_diagonal_uses_expanded_map_without_extra_corner_check(self):
        maze = np.zeros((3, 3), dtype=bool)
        maze[0, 1] = maze[1, 0] = True
        targets = np.zeros(maze.shape)
        targets[1, 1] = 1
        self.assertEqual(find_best_path_to_region(maze, (0, 0), targets), [(1, 1)])

    def test_escape_can_reach_single_pixel_safe_region(self):
        maze = np.zeros((15, 15), dtype=bool)
        danger = np.ones(maze.shape)
        danger[7, 8] = 0
        path = find_nearest_passable_path(maze, (2, 2), danger,
                                          distance_maze=np.ones(maze.shape))
        self.assertEqual(path[-1], (8, 7))

    def test_no_target_and_unreachable_target(self):
        maze = np.zeros((10, 10), dtype=bool)
        targets = np.zeros(maze.shape)
        self.assertEqual(find_best_path_to_region(maze, (1, 1), targets), [])
        targets[8, 8] = 1
        maze[5, :] = True
        self.assertEqual(find_best_path_to_region(maze, (1, 1), targets), [])

    def test_escape_reduces_danger_when_safe_area_is_unreachable(self):
        maze = np.zeros((10, 10), dtype=bool)
        maze[:, 5] = True
        danger = np.tile(np.arange(10, 0, -1), (10, 1))
        danger[:, 8:] = 0
        path = find_nearest_passable_path(maze, (1, 5), danger,
                                          distance_maze=np.ones(maze.shape))
        self.assertTrue(path)
        self.assertLess(danger[path[-1][1], path[-1][0]], danger[5, 1])
        self.assertTrue(all(x < 5 for x, _ in path))

    def test_already_at_target_stays(self):
        maze = np.zeros((10, 10), dtype=bool)
        targets = np.zeros(maze.shape)
        targets[2, 2] = 0.9
        self.assertEqual(find_best_path_to_region(maze, (2, 2), targets), [])

    def test_enemy_overlap_without_bullets_escapes_and_preserves_cache(self):
        for level in (1, 4):
            with self.subTest(level=level):
                game_map = Map.__new__(Map)
                game_map.level = level
                shape = (100, 120)
                maze = np.zeros(shape, dtype=bool)
                yy, xx = np.indices(shape)
                danger = np.maximum(0, 45 - abs(xx - 60) - abs(yy - 50))
                empty = np.zeros(shape)
                game_map.flee_mazes = (maze, np.full(shape, 10), empty, danger,
                                       empty, empty, empty, empty)
                player = SimpleNamespace(x=60, y=50)
                for _ in range(2):
                    direction, path, index = game_map.get_shot_place_direction(player, -1, None)
                    self.assertIsNotNone(direction)
                    self.assertEqual(index, 1)
                    self.assertEqual(danger[path[-1][1], path[-1][0]], 0)
                self.assertFalse(maze.any())

    def test_attack_mask_does_not_persist_into_next_escape(self):
        game_map = Map.__new__(Map)
        game_map.level = 4
        shape = (20, 30)
        maze = np.zeros(shape, dtype=bool)
        empty = np.zeros(shape)
        danger = empty.copy()
        danger[8:13, 8:13] = 10
        targets = empty.copy()
        targets[2, 20] = 0.9
        game_map.flee_mazes = (maze, np.ones(shape), empty, danger,
                               empty, empty, empty, targets)
        game_map.get_shot_place_direction(SimpleNamespace(x=2, y=2), -1, None)
        self.assertFalse(maze.any())
        direction, path, index = game_map.get_shot_place_direction(
            SimpleNamespace(x=10, y=10), -1, None)
        self.assertIsNotNone(direction)
        self.assertEqual(index, 1)
        self.assertEqual(danger[path[-1][1], path[-1][0]], 0)


if __name__ == "__main__":
    unittest.main()
