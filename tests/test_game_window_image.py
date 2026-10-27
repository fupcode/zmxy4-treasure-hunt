"""验证离线图片复用正式感知流程，不创建屏幕采集器。"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.game_window import GameWindow


class GameWindowImageTests(unittest.TestCase):

    def test_winrt_fallback_uses_on_demand_capture(self):
        camera = Mock()
        with patch("zaomeng_hanghai.game_window.dxcam.create",
                   side_effect=[RuntimeError("DXGI unsupported"), camera]) as create, \
             patch("zaomeng_hanghai.game_window.OnnxModel"), \
             patch("builtins.print"):
            window = GameWindow()
        self.assertIs(window.camera, camera)
        create.assert_called_with(backend="winrt")
        camera.start.assert_not_called()

    def test_winrt_failure_is_propagated(self):
        failure = RuntimeError("WinRT unavailable")
        with patch("zaomeng_hanghai.game_window.dxcam.create", side_effect=[RuntimeError("DXGI unsupported"), failure]), \
             patch("builtins.print"):
            with self.assertRaises(RuntimeError) as caught:
                GameWindow()
        self.assertIs(caught.exception, failure)

    def test_capture_timeout_and_invalid_region(self):
        with patch("zaomeng_hanghai.game_window.OnnxModel"):
            window = GameWindow(capture_enabled=False)
        window.camera = Mock()
        window.camera.grab.return_value = None
        with patch("zaomeng_hanghai.game_window.time.monotonic", side_effect=[0.0, 5.0]):
            with self.assertRaisesRegex(TimeoutError, "No screen frame"):
                window.capture_screen()
        window.camera.grab.side_effect = ValueError("Invalid Region")
        with self.assertRaisesRegex(ValueError, "Invalid Region"):
            window.capture_screen((-1, 0, 5, 5))
        window.camera.grab.assert_called_with(region=(-1, 0, 5, 5))

    def test_capture_region_retries_until_frame_available(self):
        with patch("zaomeng_hanghai.game_window.OnnxModel"):
            window = GameWindow(capture_enabled=False)
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        window.camera = Mock()
        window.camera.grab.side_effect = [None, image]
        with patch("zaomeng_hanghai.game_window.time.monotonic", return_value=0.0), \
             patch("zaomeng_hanghai.game_window.time.sleep"):
            result = window.capture_screen([100, 200, 105, 204])
        self.assertIs(result, image)
        self.assertEqual(window.camera.grab.call_count, 2)
        for call in window.camera.grab.call_args_list:
            self.assertEqual(call.kwargs, {"region": (100, 200, 105, 204)})

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
        self.assertEqual(window.detections["enemy"], [box])
        self.assertEqual(model.call_args.kwargs["conf_threshold"], 0.7)
        self.assertFalse(np.shares_memory(image, model.call_args.args[0]))
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
        window.detections = {"enemy": [object()]}
        with patch.object(window, "locate_map_contours", return_value=[]), \
             patch.object(window, "get_hit_point", return_value=0), \
             patch.object(window, "is_failed", return_value=False), \
             patch.object(window, "is_successful", return_value=False):
            self.assertEqual(window.update_from_image(image), -1)
            self.assertEqual(window.detections, {})
            model.assert_not_called()
            self.assertEqual(window.update_from_image(image, check_status=False), 0)
            model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
