"""用模拟时间验证 FPS 统计和十秒日志节流。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from zmxy4_treasure_hunt.voyage.env import FrameRateCalculator, Voyager


class FrameRateTests(unittest.TestCase):
    def test_terminal_reports_only_every_ten_seconds(self):
        calculator = FrameRateCalculator()
        with patch("zmxy4_treasure_hunt.voyage.env.time.perf_counter", side_effect=[i / 10 for i in range(201)]), \
             patch("builtins.print") as output:
            for _ in range(100):
                calculator.trigger(log_to_terminal=True)
            output.assert_not_called()
            calculator.trigger(log_to_terminal=True)
            self.assertEqual(output.call_count, 1)
            self.assertIn("[战斗 FPS] 10.0", output.call_args.args[0])
            self.assertIn("100.0 ms", output.call_args.args[0])
            for _ in range(100):
                calculator.trigger(log_to_terminal=True)
            self.assertEqual(output.call_count, 2)

    def test_debug_mode_updates_fps_without_terminal_output(self):
        calculator = FrameRateCalculator()
        with patch("zmxy4_treasure_hunt.voyage.env.time.perf_counter", side_effect=[0, 0.1, 10, 10.2]), \
             patch("builtins.print") as output:
            calculator.trigger()
            self.assertAlmostEqual(calculator.trigger(), 10)
            calculator.trigger()
            self.assertAlmostEqual(calculator.trigger(), 5)
            output.assert_not_called()
        calculator.reset()
        self.assertEqual(calculator.fps, 0)
        self.assertIsNone(calculator.last_time)
        self.assertIsNone(calculator.report_start)
        self.assertEqual(calculator.report_count, 0)

    def test_noncombat_wait_is_not_in_next_combat_report(self):
        voyager = Voyager.__new__(Voyager)
        voyager.map = SimpleNamespace(level=1)
        voyager.new_level = False
        voyager.enemy_list = [object()]
        voyager.bullets = []
        voyager.render_enabled = False
        voyager.fps_calculator = FrameRateCalculator()
        now = 0.0
        with patch("zmxy4_treasure_hunt.voyage.env.time.perf_counter", side_effect=lambda: now), \
                patch("builtins.print") as output:
            for index in range(101):
                now = index / 20
                voyager._update_combat_fps()
            # 战斗未满十秒就结束，过关等待不会触发报告。
            voyager.enemy_list = []
            now = 100.0
            voyager._update_combat_fps()
            output.assert_not_called()
            self.assertEqual(voyager.fps_calculator.fps, 0)
            voyager.enemy_list = [object()]
            for index in range(201):
                now = 100 + index / 20
                voyager._update_combat_fps()
            output.assert_called_once()
            self.assertIn("[战斗 FPS] 20.0", output.call_args.args[0])
            self.assertIn("50.0 ms", output.call_args.args[0])

    def test_harbor_and_level_entry_do_not_count_as_combat(self):
        voyager = Voyager.__new__(Voyager)
        voyager.enemy_list = [object()]
        voyager.bullets = []
        voyager.render_enabled = False
        for level, new_level in ((0, False), (1, True)):
            with self.subTest(level=level, new_level=new_level):
                voyager.map = SimpleNamespace(level=level)
                voyager.new_level = new_level
                voyager.fps_calculator = FrameRateCalculator()
                voyager.fps_calculator.trigger()
                voyager._update_combat_fps()
                self.assertIsNone(voyager.fps_calculator.last_time)
                self.assertIsNone(voyager.fps_calculator.report_start)


if __name__ == "__main__":
    unittest.main()
