"""验证坐标按 round 取整、往返误差与边界索引，不启动游戏。"""

import unittest
from unittest.mock import patch

import numpy as np

from zaomeng_hanghai.core.entities import Bullet, Enemy, FallingItem, Player
from zaomeng_hanghai.core.maze_utils import find_nearest_passable_point
from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.game_window import GameWindow


class CoordinateTests(unittest.TestCase):
    def test_entity_positions_and_enemy_bounds_round_to_nearest_pixel(self):
        maze = np.zeros((200, 320), dtype=bool)
        box = DetectionResult(1, [10.75 / 320, 20.75 / 200, 30.75 / 320, 40.75 / 200])
        enemy = Enemy(box, maze)
        self.assertEqual((enemy.x1, enemy.y1, enemy.x2, enemy.y2), (11, 31, 30, 41))
        self.assertEqual((enemy.x, enemy.y), (21, 34))
        player = Player(box, maze)
        self.assertEqual((player.x, player.y), (21, 35))
        bullet = Bullet((10.75 / 320, 20.75 / 200, 0))
        item = FallingItem((10.75 / 320, 20.75 / 200))
        self.assertEqual((bullet.x, bullet.y), (11, 21))
        self.assertEqual((item.x, item.y), (11, 21))
        self.assertLessEqual(abs(enemy.x1 - enemy.x1n * 320), 0.5)
        self.assertLessEqual(abs(enemy.y1 - enemy.y1n * 200), 0.5)

    def test_boundary_points_are_clamped_but_box_slice_end_can_equal_size(self):
        maze = np.zeros((200, 320), dtype=bool)
        box = DetectionResult(1, [0.999, 0.999, 1.0, 1.0])
        for entity in (Enemy(box, maze), Player(box, maze), Bullet((1, 1, 0)), FallingItem((1, 1))):
            self.assertEqual((entity.x, entity.y), (319, 199))
        enemy = Enemy(box, maze)
        self.assertEqual((enemy.x2, enemy.y2), (319, 200))  # 当前敌人框有水平收缩。
        self.assertEqual(find_nearest_passable_point(maze, (320, 200)), (319, 199))
        self.assertEqual(find_nearest_passable_point(maze, (-1, -1)), (0, 0))

    def test_image_crops_and_screen_coordinates_use_same_rounding(self):
        window = GameWindow.__new__(GameWindow)
        image = np.arange(100 * 100 * 3).reshape(100, 100, 3)
        crop = window.crop_by_rate(image, (0.106, 0.207, 0.106, 0.207))
        np.testing.assert_array_equal(crop, image[11:21, 11:21])
        window.window_box = (10, 20, 110, 120)
        window.window_size = (100, 100)
        with patch("zaomeng_hanghai.game_window.click") as click, \
             patch("zaomeng_hanghai.game_window.set_cursor_pos") as move:
            window.click_normal(0.106, 0.207)
            window.set_cursor_pos_normal(0.106, 0.207)
        click.assert_called_once_with(21, 41)
        move.assert_called_once_with(21, 41)


if __name__ == "__main__":
    unittest.main()
