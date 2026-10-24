"""关闭调试渲染时跳过画面计算，同时保留寻路数据。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from zaomeng_hanghai.core.entities import Bullet
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.env import Env


class RenderConfigTests(unittest.TestCase):
    def test_disabled_skips_window_render_but_keeps_fps_counter(self):
        with patch("zaomeng_hanghai.env.GameWindow") as window_class, \
             patch("zaomeng_hanghai.env.FrameRateCalculator") as fps_class, \
             patch("zaomeng_hanghai.env.click") as click, \
             patch("zaomeng_hanghai.env.cv2.imshow") as imshow, \
             patch("zaomeng_hanghai.env.cv2.moveWindow") as move_window, \
             patch("zaomeng_hanghai.env.cv2.setWindowProperty") as set_property, \
             patch("zaomeng_hanghai.env.cv2.waitKey") as wait_key:
            window_class.return_value.window_box = (0, 0, 800, 600)
            env = Env(render_enabled=False)
            self.assertFalse(env.map.render_enabled)
            env.init_window()
            with patch("zaomeng_hanghai.env.np.zeros", side_effect=AssertionError("不应创建渲染画布")):
                env.render([(1, 1), (2, 2)])
            window_class.return_value.init_window.assert_called_once_with(None)
            click.assert_called_once_with(790, 590)
            fps_class.assert_called_once_with()
            fps_class.return_value.trigger.assert_not_called()
            for operation in (imshow, move_window, set_property, wait_key):
                operation.assert_not_called()

    def test_disabled_skips_debug_image_but_preserves_danger_maps(self):
        enabled = Map()
        disabled = Map(render_enabled=False)
        for game_map in (enabled, disabled):
            game_map.reset(0)
        player = SimpleNamespace(x=100, y=100)
        bullets = [Bullet((0.6, 0.6, 2))]
        expected = enabled.get_flee_maze(player, bullets, [])
        with patch("zaomeng_hanghai.core.map.np.stack", side_effect=AssertionError("不应生成调试图")):
            actual = disabled.get_flee_maze(player, bullets, [])
        self.assertIsNotNone(enabled.debug_image)
        self.assertIsNone(disabled.debug_image)
        for expected_array, actual_array in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(actual_array, expected_array)


if __name__ == "__main__":
    unittest.main()
