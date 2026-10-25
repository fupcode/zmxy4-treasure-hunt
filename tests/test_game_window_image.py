"""验证离线图片复用正式感知流程，不创建屏幕采集器。"""

import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.game_window import GameWindow


class GameWindowImageTests(unittest.TestCase):
    def test_default_backend_is_preferred(self):
        camera = Mock()
        with patch("zaomeng_hanghai.game_window.dxcam.create", return_value=camera) as create, \
             patch("zaomeng_hanghai.game_window.OnnxModel"):
            window = GameWindow()
        create.assert_called_once_with()
        self.assertIs(window.camera, camera)
        camera.start.assert_called_once_with(target_fps=60)

    def test_falls_back_to_winrt_when_default_initialization_fails(self):
        camera = Mock()
        with patch("zaomeng_hanghai.game_window.dxcam.create", side_effect=[RuntimeError("DXGI unsupported"), camera]) as create, \
             patch("zaomeng_hanghai.game_window.OnnxModel"), \
             patch("builtins.print"):
            window = GameWindow()
        self.assertEqual(create.call_args_list, [call(), call(backend="winrt")])
        self.assertIs(window.camera, camera)
        camera.start.assert_called_once_with(target_fps=60)

    def test_winrt_failure_is_propagated(self):
        failure = RuntimeError("WinRT unavailable")
        with patch("zaomeng_hanghai.game_window.dxcam.create", side_effect=[RuntimeError("DXGI unsupported"), failure]), \
             patch("builtins.print"):
            with self.assertRaises(RuntimeError) as caught:
                GameWindow()
        self.assertIs(caught.exception, failure)

    def test_background_capture_reads_latest_full_frame_and_crops(self):
        camera = Mock()
        first = np.zeros((8, 10, 3), dtype=np.uint8)
        latest = np.full((8, 10, 3), 123, dtype=np.uint8)
        camera.grab.side_effect = [None, first, latest]
        with patch("zaomeng_hanghai.game_window.dxcam.create", return_value=camera), \
             patch("zaomeng_hanghai.game_window.OnnxModel"), \
             patch("zaomeng_hanghai.game_window.time.sleep"):
            window = GameWindow()
            np.testing.assert_array_equal(window.capture_screen(), first)
            cropped = window.capture_screen((2, 1, 7, 6))
        camera.start.assert_called_once_with(target_fps=60)
        self.assertTrue(all(not call.args and not call.kwargs for call in camera.grab.call_args_list))
        np.testing.assert_array_equal(cropped, latest[1:6, 2:7])
        self.assertFalse(np.shares_memory(cropped, latest))

    def test_capture_timeout_and_invalid_region(self):
        with patch("zaomeng_hanghai.game_window.OnnxModel"):
            window = GameWindow(capture_enabled=False)
        window.camera = Mock()
        window.camera.grab.return_value = None
        with patch("zaomeng_hanghai.game_window.time.monotonic", side_effect=[0.0, 5.0]):
            with self.assertRaisesRegex(TimeoutError, "No screen frame"):
                window.capture_screen()
        window.camera.grab.return_value = np.zeros((8, 10, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "outside frame size"):
            window.capture_screen((-1, 0, 5, 5))

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
