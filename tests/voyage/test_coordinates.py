"""验证坐标按 round 取整、往返误差与边界索引，不启动游戏。"""

import unittest
from unittest.mock import patch

import numpy as np

from zmxy4_treasure_hunt.voyage.game_window import GameWindow, GameWindowError


class CoordinateTests(unittest.TestCase):

    def test_window_initialization_focuses_top_left_after_coordinate_offset(self):
        for outer_box in (None, (100, 200, 500, 600)):
            for focus_game in (True, False):
                with self.subTest(outer_box=outer_box, focus_game=focus_game):
                    window = GameWindow.__new__(GameWindow)
                    with (
                        patch.object(window, "capture_screen", return_value=np.zeros((100, 120, 3), dtype=np.uint8)),
                        patch.object(window, "locate_window", return_value=[5, 6, 105, 86]),
                        patch("zmxy4_treasure_hunt.voyage.game_window.click") as click,
                        patch("builtins.print"),
                    ):
                        window.init_window(outer_box, focus_game=focus_game)
                    self.assertEqual(window.window_size, (100, 80))
                    if focus_game:
                        expected = (115, 216) if outer_box else (15, 16)
                        click.assert_called_once_with(*expected)
                    else:
                        click.assert_not_called()

    def test_uninitialized_geometry_rejects_mouse_actions(self):
        window = GameWindow.__new__(GameWindow)
        with patch("zmxy4_treasure_hunt.voyage.game_window.click") as click, \
             patch("zmxy4_treasure_hunt.voyage.game_window.set_cursor_pos") as move:
            for box, size in ((None, None), ((10, 20, 110, 120), None),
                              (None, (100, 100))):
                window.window_box, window.window_size = box, size
                with self.subTest(box=box, size=size):
                    for action in (window.click_normal, window.set_cursor_pos_normal):
                        with self.assertRaisesRegex(GameWindowError, "init_window"):
                            action(0.5, 0.5)
            click.assert_not_called()
            move.assert_not_called()

    def test_image_crops_and_screen_coordinates_use_same_rounding(self):
        window = GameWindow.__new__(GameWindow)
        image = np.arange(100 * 100 * 3).reshape(100, 100, 3)
        crop = window.crop_by_rate(image, (0.106, 0.207, 0.106, 0.207))
        np.testing.assert_array_equal(crop, image[11:21, 11:21])
        window.window_box = (10, 20, 110, 120)
        window.window_size = (100, 100)
        with patch("zmxy4_treasure_hunt.voyage.game_window.click") as click, \
             patch("zmxy4_treasure_hunt.voyage.game_window.set_cursor_pos") as move:
            window.click_normal(0.106, 0.207)
            window.set_cursor_pos_normal(0.106, 0.207)
        click.assert_called_once_with(21, 41)
        move.assert_called_once_with(21, 41)


if __name__ == "__main__":
    unittest.main()
