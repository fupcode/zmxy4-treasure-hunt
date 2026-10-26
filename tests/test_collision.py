"""静态地图碰撞与动作方向检查，不启动游戏。"""

from types import SimpleNamespace
import unittest

import numpy as np

from zaomeng_hanghai.core.constants import MAZE_HEIGHT, MAZE_WIDTH
from zaomeng_hanghai.core.direction import move_list
from zaomeng_hanghai.core.map import Map


def empty_map():
    game_map = Map.__new__(Map)
    game_map.level = 1
    game_map.level_mazes = [np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool) for _ in range(4)]
    game_map.map_template_without_exit = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
    return game_map


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.game_map = empty_map()
        self.player = SimpleNamespace(x=100, y=100)

    def test_stop_and_free_movement(self):
        for action in range(9):
            self.assertFalse(self.game_map.is_move_collision(self.player, action))
        self.game_map.get_maze()[99, 99] = True
        self.assertFalse(self.game_map.is_move_collision(self.player, 0))

    def test_thin_wall_and_diagonal_without_side_checks(self):
        self.game_map.get_maze()[100, 102] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 1))
        self.assertFalse(self.game_map.is_move_collision(self.player, 3))
        self.game_map.get_maze()[100, 101] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 1))
        self.assertFalse(self.game_map.is_move_collision(self.player, 5))
        self.game_map.get_maze()[100, 101] = False
        self.game_map.get_maze()[101, 100] = True
        self.assertFalse(self.game_map.is_move_collision(self.player, 5))

    def test_bounds_and_common_forbidden_region(self):
        for x, y, action in ((0, 100, 3), (319, 100, 1), (100, 0, 4), (100, 199, 2)):
            self.assertTrue(self.game_map.is_move_collision(SimpleNamespace(x=x, y=y), action))
        self.game_map.map_template_without_exit[100, 101] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 1))
        for action in (-1, 9, 0.5):
            with self.assertRaises(ValueError):
                self.game_map.is_move_collision(self.player, action)

    def test_all_directions_check_three_pixels(self):
        maze = self.game_map.get_maze()
        for direction, (dx, dy) in enumerate(move_list, 1):
            for distance in (1, 2, 3):
                maze.fill(False)
                maze[100 + dy * distance, 100 + dx * distance] = True
                self.assertTrue(self.game_map.is_move_collision(self.player, direction))
            maze.fill(False)
            maze[100 + dy * 4, 100 + dx * 4] = True
            self.assertFalse(self.game_map.is_move_collision(self.player, direction))
        maze.fill(False)
        maze[102, 101] = True
        self.assertFalse(self.game_map.is_move_collision(self.player, 5))


if __name__ == "__main__":
    unittest.main()
