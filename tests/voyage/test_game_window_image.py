"""验证离线图片复用正式感知流程，不创建屏幕采集器。"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from zmxy4_treasure_hunt.voyage.core.onnx_model import DetectionResult
from zmxy4_treasure_hunt.voyage.game_window import GameWindow


class GameWindowImageTests(unittest.TestCase):

    def test_permission_failure_prevents_initial_click(self):
        with patch('zmxy4_treasure_hunt.voyage.game_window.OnnxModel'):
            window = GameWindow(capture_enabled=False)
        with patch.object(window, 'capture_screen', return_value=np.zeros((400, 600, 3), dtype=np.uint8)), \
             patch.object(window, 'locate_window', return_value=[10, 20, 610, 420]), \
             patch('zmxy4_treasure_hunt.voyage.game_window.ensure_window_permission', side_effect=PermissionError('权限不足')) as check, \
             patch('zmxy4_treasure_hunt.voyage.game_window.click') as click:
            with self.assertRaises(PermissionError):
                window.init_window((100, 200, 1000, 900))
        check.assert_called_once_with((410, 420))
        click.assert_not_called()

    def test_winrt_fallback_uses_on_demand_capture(self):
        camera = Mock()
        with patch("zmxy4_treasure_hunt.voyage.game_window.dxcam.create",
                   side_effect=[RuntimeError("DXGI unsupported"), camera]) as create, \
             patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel"), \
             patch("builtins.print"):
            window = GameWindow()
        self.assertIs(window.camera, camera)
        create.assert_called_with(backend="winrt")
        camera.start.assert_not_called()

    def test_winrt_failure_is_propagated(self):
        failure = RuntimeError("WinRT unavailable")
        with patch("zmxy4_treasure_hunt.voyage.game_window.dxcam.create", side_effect=[RuntimeError("DXGI unsupported"), failure]), \
             patch("builtins.print"):
            with self.assertRaises(RuntimeError) as caught:
                GameWindow()
        self.assertIs(caught.exception, failure)

    def test_capture_timeout_and_invalid_region(self):
        with patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel"):
            window = GameWindow(capture_enabled=False)
        window.camera = Mock()
        window.camera.grab.return_value = None
        with patch("zmxy4_treasure_hunt.voyage.game_window.time.monotonic", side_effect=[0.0, 5.0]):
            with self.assertRaisesRegex(TimeoutError, "No screen frame"):
                window.capture_screen()
        window.camera.grab.side_effect = ValueError("Invalid Region")
        with self.assertRaisesRegex(ValueError, "Invalid Region"):
            window.capture_screen((-1, 0, 5, 5))
        window.camera.grab.assert_called_with(region=(-1, 0, 5, 5))

    def test_capture_region_retries_until_frame_available(self):
        with patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel"):
            window = GameWindow(capture_enabled=False)
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        window.camera = Mock()
        window.camera.grab.side_effect = [None, image]
        with patch("zmxy4_treasure_hunt.voyage.game_window.time.monotonic", return_value=0.0), \
             patch("zmxy4_treasure_hunt.voyage.game_window.time.sleep"):
            result = window.capture_screen([100, 200, 105, 204])
        self.assertIs(result, image)
        self.assertEqual(window.camera.grab.call_count, 2)
        for call in window.camera.grab.call_args_list:
            self.assertEqual(call.kwargs, {"region": (100, 200, 105, 204)})

    def test_processing_masks_input_in_place_and_uses_threshold(self):
        box = DetectionResult(0.9, [0.2, 0.3, 0.4, 0.6])
        model = Mock(return_value={
            "enemy": [box], "player": [box], "bullet": [],
            "pao": [], "dan": [], "fan": [], "jia": [],
        })
        with patch("zmxy4_treasure_hunt.voyage.game_window.dxcam.create") as capture, \
             patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel", return_value=model):
            window = GameWindow(capture_enabled=False)
        capture.assert_not_called()
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        with patch.object(window, "locate_map_contours", return_value=[object()]), \
             patch.object(window, "get_hit_point", return_value=4):
            self.assertEqual(window.update_from_image(image, conf_threshold=0.7), 0)
        expected = np.zeros_like(image)
        expected[:28, :25] = 255
        expected[174:, :104] = 255
        np.testing.assert_array_equal(image, expected)
        self.assertEqual(window.enemy_boxes, [box])
        self.assertEqual(window.detections["enemy"], [box])
        self.assertEqual(model.call_args.kwargs["conf_threshold"], 0.7)
        self.assertIs(image, model.call_args.args[0])
        with self.assertRaisesRegex(RuntimeError, "capture is disabled"):
            window.capture_screen()

    def test_first_missing_player_retries_then_reuses_last_detection(self):
        box = DetectionResult(0.9, [0.2, 0.3, 0.4, 0.6])
        detections = {
            "enemy": [], "player": [], "bullet": [],
            "pao": [], "dan": [], "fan": [], "jia": [],
        }
        with patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel",
                   return_value=Mock(return_value=detections)):
            window = GameWindow(capture_enabled=False)
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        with (
            patch.object(window, "locate_map_contours", return_value=[object()]),
            patch.object(window, "get_hit_point", return_value=4),
        ):
            self.assertEqual(window.update_from_image(image), -1)
            self.assertIsNone(window.player_box)
            detections["player"] = [box]
            self.assertEqual(window.update_from_image(image), 0)
            self.assertIs(window.player_box, box)
            detections["player"] = []
            self.assertEqual(window.update_from_image(image), 0)
            self.assertIs(window.player_box, box)

    def test_status_gate_and_debug_override(self):
        model = Mock(return_value={
            "enemy": [], "player": [], "bullet": [],
            "pao": [], "dan": [], "fan": [], "jia": [],
        })
        with patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel", return_value=model):
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
