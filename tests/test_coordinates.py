"""验证坐标按 round 取整、往返误差与边界索引，不启动游戏。"""

import unittest
from unittest.mock import patch

import numpy as np

from zaomeng_hanghai.game_window import GameWindow


class CoordinateTests(unittest.TestCase):

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
