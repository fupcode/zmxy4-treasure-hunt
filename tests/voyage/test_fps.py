"""用模拟时间验证 FPS 统计和十秒日志节流。"""

import unittest
from unittest.mock import patch

from zmxy4_treasure_hunt.voyage.env import FrameRateCalculator


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
            self.assertIn("[FPS] 10.0", output.call_args.args[0])
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


if __name__ == "__main__":
    unittest.main()
