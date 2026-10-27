"""自动战斗寻路的窄目标、墙角和敌人重叠回归测试。"""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.constants import (
    MAZE_HEIGHT, MAZE_WIDTH, BULLET_AVOID_HEIGHT, BULLET_AVOID_WIDTH,
    FIRE_X_RANGE, FIRE_Y_RANGE, HIT_HEIGHT, HIT_WIDTH,
)
from zaomeng_hanghai.core.entities import Bullets
from zaomeng_hanghai.core.direction import Direction
from zaomeng_hanghai.core.maze_utils import (
    find_best_path_to_region,
    find_nearest_passable_path,
)


class NavigationTests(unittest.TestCase):

    def test_shooting_path_uses_last_executed_direction(self):
        game_map = Map.__new__(Map)
        game_map.level = 1
        shape = (25, 30)
        empty = np.zeros(shape)
        targets = empty.copy()
        targets[12, 2] = targets[12, 22] = 1
        game_map.flee_mazes = (empty.astype(bool), np.ones(shape), empty, empty,
                               empty, empty, empty, targets)
        direction, path, mode = game_map.get_shot_place_direction(
            SimpleNamespace(x=12, y=12), last_direction=Direction.Right)
        self.assertEqual(direction, Direction.Right)
        self.assertEqual(path[-1], (22, 12))
        self.assertEqual(mode, 4)

    def test_bullet_rectangles_clip_at_edges_with_hit_tail(self):
        game_map = Map.__new__(Map)
        game_map.level = 1
        game_map.render_enabled = False
        shape = (MAZE_HEIGHT, MAZE_WIDTH)
        game_map.get_maze = lambda: np.zeros(shape, dtype=bool)
        game_map.get_distance_maze = lambda: np.ones(shape)
        game_map.map_template_without_exit = np.zeros(shape, dtype=bool)
        player = SimpleNamespace(x=160, y=100)
        for direction in range(4):
            for x, y in ((0, 0), (319, 0), (0, 199), (319, 199), (160, 100)):
                with self.subTest(direction=direction, x=x, y=y):
                    bullet = SimpleNamespace(x=x, y=y, d=direction)
                    regions = game_map.get_flee_maze(player, [bullet], [])
                    danger, blocked, distances = regions[2], regions[0], regions[1]
                    self.assertGreater(danger[y, x], 0)
                    player_inside = (abs(player.x - x) <= BULLET_AVOID_WIDTH
                                     and abs(player.y - y) <= BULLET_AVOID_HEIGHT)
                    self.assertEqual(bool(blocked[y, x]), not player_inside)
                    self.assertEqual(distances[y, x], 0)
                    left, up, right, down = game_map.get_bullet_danger_bounds(bullet)
                    expected = np.zeros(shape, dtype=bool)
                    expected[up:down, left:right] = True
                    np.testing.assert_array_equal(danger > 0, expected)
        bullet = SimpleNamespace(x=160, y=100, d=0)
        danger = game_map.get_flee_maze(player, [bullet], [])[2]
        self.assertGreater(danger[100 + BULLET_AVOID_HEIGHT, 160], 0)
        self.assertGreater(danger[100, 160 - HIT_WIDTH], 0)
        self.assertEqual(danger[100, 160 - HIT_WIDTH - 1], 0)
        self.assertGreater(danger[100, 160 + FIRE_X_RANGE], 0)
        self.assertEqual(danger[100 + BULLET_AVOID_HEIGHT + 1, 160], 0)
        bullet.d = 1
        danger = game_map.get_flee_maze(player, [bullet], [])[2]
        self.assertGreater(danger[100, 160 + BULLET_AVOID_WIDTH], 0)
        self.assertGreater(danger[100 - HIT_HEIGHT, 160], 0)
        self.assertEqual(danger[100 - HIT_HEIGHT - 1, 160], 0)
        forward_y = min(MAZE_HEIGHT - 1, 100 + FIRE_Y_RANGE)
        self.assertGreater(danger[forward_y, 160], 0)
        bullet.d = 2
        danger = game_map.get_flee_maze(player, [bullet], [])[2]
        self.assertGreater(danger[100, 160 + HIT_WIDTH], 0)
        self.assertEqual(danger[100, 160 + HIT_WIDTH + 1], 0)
        self.assertGreater(danger[100, 160 - FIRE_X_RANGE], 0)
        bullet.d = 3
        danger = game_map.get_flee_maze(player, [bullet], [])[2]
        self.assertGreater(danger[100 + HIT_HEIGHT, 160], 0)
        self.assertEqual(danger[100 + HIT_HEIGHT + 1, 160], 0)
        self.assertGreater(danger[max(0, 100 - FIRE_Y_RANGE), 160], 0)

    def test_overlapping_player_bullets_remain_costs_and_allow_escape(self):
        game_map = Map.__new__(Map)
        game_map.level = 1
        game_map.render_enabled = False
        shape = (MAZE_HEIGHT, MAZE_WIDTH)
        maze = np.zeros(shape, dtype=bool)
        maze[125, 45] = True
        game_map.get_maze = lambda: maze
        game_map.get_distance_maze = lambda: np.ones(shape)
        game_map.map_template_without_exit = np.zeros(shape, dtype=bool)
        player = SimpleNamespace(x=42, y=128)
        overlapping = [SimpleNamespace(x=57, y=125, d=2),
                       SimpleNamespace(x=59, y=119, d=1)]
        other = SimpleNamespace(x=80, y=150, d=2)
        game_map.update(player, overlapping + [other], [])
        regions = game_map.flee_mazes
        self.assertFalse(regions[0][player.y, player.x])
        self.assertTrue(regions[0][125, 45])  # 静态墙仍然保留。
        self.assertTrue(regions[0][other.y, other.x])
        self.assertGreater(regions[2][player.y, player.x], 0)
        for order in (overlapping + [other], [other] + overlapping[::-1]):
            actual = game_map.get_flee_maze(player, order, [])
            np.testing.assert_array_equal(actual[0], regions[0])
            np.testing.assert_array_equal(actual[2], regions[2])
        direction, path, mode = game_map.get_shot_place_direction(player)
        self.assertIsNotNone(direction)
        self.assertEqual(mode, 1)
        self.assertTrue(path)
        self.assertEqual(regions[2][path[-1][1], path[-1][0]], 0)
        self.assertTrue(all(not regions[0][y, x] for x, y in path))
        # 玩家撤出后重新生成地图，这两颗子弹恢复为硬障碍。
        rebuilt = game_map.get_flee_maze(SimpleNamespace(x=120, y=60), overlapping, [])
        self.assertTrue(rebuilt[0][128, 42])

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
                        game_map.get_shot_place_direction(SimpleNamespace(x=x, y=y))
                    expected = maze | (((bullets + enemies) > 0)
                                       & (abs(xx - x) <= 10) & (abs(yy - y) <= 10))
                    self.assertEqual(search.call_count, 1)
                    for invocation in search.call_args_list:
                        np.testing.assert_array_equal(invocation.args[0], expected)
                        if local_peek:
                            self.assertIsNone(invocation.kwargs.get('cost_map'))
                        else:
                            np.testing.assert_array_equal(invocation.kwargs['cost_map'], 2.0 * peek)
                    self.assertEqual(np.count_nonzero(maze), 1)

    def test_local_failure_waits_without_global_fallback_including_boss(self):
        shape = (20, 30)
        for level in (1, 4):
            game_map = Map.__new__(Map)
            game_map.level = level
            empty = np.zeros(shape)
            global_targets = empty.copy()
            global_targets[10, 25] = 1
            vision = empty.copy()
            vision[:, :15] = 1
            game_map.flee_mazes = (empty.astype(bool), np.ones(shape), empty, empty,
                                   vision, empty, empty, global_targets)
            with patch('zaomeng_hanghai.core.map.maze_utils.find_best_path_to_region',
                       return_value=[]) as search:
                result = game_map.get_shot_place_direction(SimpleNamespace(x=5, y=10))
            self.assertEqual(result, (None, [], 3))
            search.assert_called_once()
            np.testing.assert_array_equal(search.call_args.args[2], empty)

    def test_outside_vision_searches_global_targets(self):
        game_map = Map.__new__(Map)
        game_map.level = 1
        shape = (20, 30)
        empty = np.zeros(shape)
        targets = empty.copy()
        targets[10, 20] = 1
        vision = empty.copy()
        vision[5:15, 10:15] = 2
        game_map.flee_mazes = (empty.astype(bool), np.ones(shape), empty, empty,
                               vision, empty, empty, targets)
        with patch('zaomeng_hanghai.core.map.maze_utils.find_best_path_to_region',
                   wraps=find_best_path_to_region) as search:
            direction, path, mode = game_map.get_shot_place_direction(
                SimpleNamespace(x=5, y=10))
        np.testing.assert_array_equal(search.call_args.kwargs['cost_map'], 2.0 * vision)
        self.assertIsNotNone(direction)
        self.assertEqual(mode, 4)
        self.assertEqual(path[-1], (20, 10))

    def test_local_target_can_be_reached_by_detouring_outside_vision(self):
        game_map = Map.__new__(Map)
        game_map.level = 1
        shape = (20, 30)
        empty = np.zeros(shape)
        maze = empty.astype(bool)
        maze[5:15, 15] = True
        vision = empty.copy()
        vision[5:15, 5:25] = 1
        targets = empty.copy()
        targets[10, 20] = 0.9
        game_map.flee_mazes = (maze, np.ones(shape), empty, empty,
                               vision, targets > 0, targets, empty)
        direction, path, mode = game_map.get_shot_place_direction(
            SimpleNamespace(x=10, y=10))
        self.assertIsNotNone(direction)
        self.assertEqual(mode, 3)
        self.assertEqual(path[-1], (20, 10))
        self.assertTrue(any(vision[y, x] == 0 for x, y in path))
        self.assertTrue(all(not maze[y, x] for x, y in path))

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

    def test_extra_cost_avoids_expensive_intermediate_cells_on_coarse_path(self):
        maze = np.zeros((25, 30), dtype=bool)
        targets = np.zeros(maze.shape)
        targets[2, 22] = 0.9
        extra = np.zeros(maze.shape)
        # x=4 不在粗搜索落脚点上，仍须逐格计费。
        extra[:10, 4] = 100
        start = (2, 2)
        direct = find_best_path_to_region(maze, start, targets, hate_factor=0)
        detour = find_best_path_to_region(maze, start, targets, hate_factor=0, cost_map=extra)
        self.assertEqual(direct[-1], (22, 2))
        self.assertEqual(detour[-1], (22, 2))
        def extra_cost(path):
            total = 0
            for (x1, y1), (x2, y2) in zip([start] + path, path):
                stride = max(abs(x2 - x1), abs(y2 - y1))
                dx, dy = (x2 - x1) // stride, (y2 - y1) // stride
                total += sum(extra[y1 + dy * step, x1 + dx * step]
                             for step in range(1, stride + 1))
            return total
        self.assertGreater(extra_cost(direct), 0)
        self.assertEqual(extra_cost(detour), 0)

    def test_extra_cost_does_not_enable_partial_escape(self):
        maze = np.zeros((15, 15), dtype=bool)
        maze[:, 7] = True
        targets = np.zeros(maze.shape)
        targets[10, 10] = 1
        extra = np.ones(maze.shape)
        extra[:, :3] = 10
        self.assertEqual(find_best_path_to_region(
            maze, (1, 1), targets, cost_map=extra), [])

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
            SimpleNamespace(x=2, y=10))
        self.assertIsNotNone(direction)
        self.assertEqual(index, 3)
        self.assertEqual(path[-1], (10, 10))
        self.assertEqual(game_map.get_shot_place_direction(
            SimpleNamespace(x=10, y=10)), (None, [], 0))

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
            SimpleNamespace(x=10, y=10)), (None, [], 0))

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
                    direction, path, index = game_map.get_shot_place_direction(player)
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
        game_map.get_shot_place_direction(SimpleNamespace(x=2, y=2))
        self.assertFalse(maze.any())
        direction, path, index = game_map.get_shot_place_direction(
            SimpleNamespace(x=10, y=10))
        self.assertIsNotNone(direction)
        self.assertEqual(index, 1)
        self.assertEqual(danger[path[-1][1], path[-1][0]], 0)


if __name__ == "__main__":
    unittest.main()
