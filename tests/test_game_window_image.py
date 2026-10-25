"""验证离线图片复用正式感知流程，不创建屏幕采集器。"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.game_window import GameWindow


class GameWindowImageTests(unittest.TestCase):
    def test_offline_processing_preserves_source_and_uses_threshold(self):
        box = DetectionResult(0.9, [0.2, 0.3, 0.4, 0.6])
        model = Mock(return_value={
            "enemy": [box], "player": [], "bullet": [],
            "pao": [], "dan": [], "fan": [], "jia": [],
        })
        with patch("zaomeng_hanghai.game_window.dxcam.create") as capture, \
             patch("zaomeng_hanghai.game_window.OnnxModel", return_value=model):
            window = GameWindow(capture_enabled=False)
        capture.assert_not_called()
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        with patch.object(window, "locate_map_contours", return_value=[object()]), \
             patch.object(window, "get_hit_point", return_value=4):
            self.assertEqual(window.update_from_image(image, conf_threshold=0.7), 0)
        np.testing.assert_array_equal(image, 0)
        self.assertEqual(window.enemy_boxes, [box])
        self.assertEqual(model.call_args.kwargs["conf_threshold"], 0.7)
        self.assertFalse(np.shares_memory(image, window.window_image))
        with self.assertRaisesRegex(RuntimeError, "capture is disabled"):
            window.capture_screen()

    def test_status_gate_and_debug_override(self):
        model = Mock(return_value={
            "enemy": [], "player": [], "bullet": [],
            "pao": [], "dan": [], "fan": [], "jia": [],
        })
        with patch("zaomeng_hanghai.game_window.OnnxModel", return_value=model):
            window = GameWindow(capture_enabled=False)
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        with patch.object(window, "locate_map_contours", return_value=[]), \
             patch.object(window, "get_hit_point", return_value=0), \
             patch.object(window, "is_failed", return_value=False), \
             patch.object(window, "is_successful", return_value=False):
            self.assertEqual(window.update_from_image(image), -1)
            model.assert_not_called()
            self.assertEqual(window.update_from_image(image, check_status=False), 0)
            model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
