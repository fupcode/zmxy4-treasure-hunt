"""专家战斗寻路的窄目标、墙角和敌人重叠回归测试。"""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.constants import MAZE_HEIGHT, MAZE_WIDTH
from zaomeng_hanghai.core.maze_utils import (
    find_best_path_to_region,
    find_nearest_passable_path,
)


class ExpertNavigationTests(unittest.TestCase):
    def test_shooting_blocks_only_danger_in_ten_cell_square(self):
        shape = (40, 50)
        for start in ((20, 20), (0, 0), (49, 39)):
            for local_peek in (False, True):
                with self.subTest(start=start, local_peek=local_peek):
                    game_map = Map.__new__(Map)
                    game_map.level = 1
                    maze = np.zeros(shape, dtype=bool)
                    maze[5, 5] = True
                    yy, xx = np.indices(shape)
                    # 子弹和敌人危险交错分布，并留出安全格子。
                    bullets = ((xx + yy) % 3 == 0).astype(np.int64)
                    enemies = ((xx + yy) % 3 == 1).astype(np.int64)
                    x, y = start
                    bullets[y, x] = enemies[y, x] = 0
                    empty = np.zeros(shape)
                    peek = np.ones(shape) if local_peek else empty
                    game_map.flee_mazes = (maze, np.ones(shape), bullets, enemies,
                                           peek, empty, empty, empty)
                    with patch("zaomeng_hanghai.core.map.maze_utils.find_best_path_to_region", return_value=[]) as search:
                        game_map.get_shot_place_direction(SimpleNamespace(x=x, y=y), None, None)
                    expected = maze | (((bullets + enemies) > 0)
                                       & (abs(xx - x) <= 10) & (abs(yy - y) <= 10))
                    self.assertEqual(search.call_count, 2 if local_peek else 1)
                    for invocation in search.call_args_list:
                        np.testing.assert_array_equal(invocation.args[0], expected)
                    self.assertEqual(np.count_nonzero(maze), 1)

    def test_overlapping_bullet_danger_adds_and_is_order_independent(self):
        shape = (MAZE_HEIGHT, MAZE_WIDTH)
        game_map = Map.__new__(Map)
        game_map.render_enabled = False
        game_map.map_template_without_exit = np.zeros(shape, dtype=bool)
        game_map.get_maze = lambda: np.zeros(shape, dtype=bool)
        game_map.get_distance_maze = lambda: np.ones(shape)
        player = SimpleNamespace(x=160, y=100)

        def danger(bullets):
            return game_map.get_flee_maze(player, bullets, [])[2]

        for level in (1, 4):
            game_map.level = level
            for direction in range(4):
                with self.subTest(level=level, direction=direction):
                    first = SimpleNamespace(x=160, y=100, d=direction)
                    second = SimpleNamespace(x=163, y=103, d=direction)
                    first_danger, second_danger = danger([first]), danger([second])
                    overlap = (first_danger > 0) & (second_danger > 0)
                    self.assertTrue(overlap.any())
                    combined = danger([first, second])
                    np.testing.assert_array_equal(combined, first_danger + second_danger)
                    np.testing.assert_array_equal(combined, danger([second, first]))
            bullets = [SimpleNamespace(x=160, y=100, d=direction) for direction in range(4)]
            expected = sum((danger([bullet]) for bullet in bullets), np.zeros(shape, dtype=np.int64))
            np.testing.assert_array_equal(danger(bullets), expected)
            np.testing.assert_array_equal(danger(bullets[::-1]), expected)

    def test_coarse_search_is_used_when_target_is_on_five_pixel_grid(self):
        maze = np.zeros((25, 30), dtype=bool)
        targets = np.zeros(maze.shape)
        targets[12, 22] = 0.9
        path = find_best_path_to_region(maze, (2, 2), targets,
                                        distance_maze=np.ones(maze.shape))
        self.assertEqual(path[-1], (22, 12))
        points = [(2, 2)] + path
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            self.assertEqual(max(abs(x2 - x1), abs(y2 - y1)), 5)

    def test_coarse_search_cannot_jump_through_thin_wall(self):
        maze = np.zeros((20, 30), dtype=bool)
        maze[:, 9] = True
        targets = np.zeros(maze.shape)
        targets[12, 22] = 0.9
        self.assertEqual(find_best_path_to_region(
            maze, (2, 2), targets, distance_maze=np.ones(maze.shape)), [])

    def test_coarse_partial_escape_does_not_skip_fine_search(self):
        maze = np.zeros((16, 16), dtype=bool)
        danger = np.full(maze.shape, 10.0)
        danger[:, 6:] = 5
        danger[8, 9] = 0
        path = find_nearest_passable_path(maze, (1, 1), danger,
                                          distance_maze=np.ones(maze.shape))
        self.assertEqual(path[-1], (9, 8))
        points = [(1, 1)] + path
        self.assertTrue(all(max(abs(x2-x1), abs(y2-y1)) == 1
                            for (x1, y1), (x2, y2) in zip(points, points[1:])))

    def test_peek_approaches_alignment_then_stays_instead_of_retreating(self):
        game_map = Map.__new__(Map)
        game_map.level = 1
        shape = (20, 30)
        maze = np.zeros(shape, dtype=bool)
        empty = np.zeros(shape)
        peek = np.ones(shape)
        cross = empty.copy()
        cross[:, 10] = 1
        targets = empty.copy()
        targets[10, 10] = 0.9
        game_map.flee_mazes = (maze, np.ones(shape), empty, empty,
                               peek, cross, targets, targets)
        direction, path, index = game_map.get_shot_place_direction(
            SimpleNamespace(x=2, y=10), None, None)
        self.assertIsNotNone(direction)
        self.assertEqual(index, 3)
        self.assertEqual(path[-1], (10, 10))
        self.assertEqual(game_map.get_shot_place_direction(
            SimpleNamespace(x=10, y=10), index, path[-1]), (None, [], 0))

    def test_valid_shooting_position_does_not_chase_higher_peek_score(self):
        game_map = Map.__new__(Map)
        game_map.level = 1
        shape = (20, 30)
        maze = np.zeros(shape, dtype=bool)
        empty = np.zeros(shape)
        targets = empty.copy()
        targets[10, 10] = 0.4
        targets[10, 15] = 0.9
        game_map.flee_mazes = (maze, np.ones(shape), empty, empty,
                               np.ones(shape), targets > 0, targets, targets)
        self.assertEqual(game_map.get_shot_place_direction(
            SimpleNamespace(x=10, y=10), None, None), (None, [], 0))

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
