import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from zmxy4_treasure_hunt.click_capture import ClickCapture


class ClickCaptureTests(unittest.TestCase):
    def setUp(self):
        self.capture = ClickCapture()
        self.capture.root = Mock()
        self.capture.canvas = Mock()
        self.capture.origin = (-100, -50)
        self.capture.screenshot = SimpleNamespace(width=1000, height=800)
        self.capture.hint = 1
        self.capture.help_text = 2
        self.capture.show_preview = Mock(side_effect=self.set_preview)

    def set_preview(self, region):
        self.capture.preview_region = region

    def event(self, x, y):
        return SimpleNamespace(x_root=x, y_root=y)

    def test_automatic_click_confirms_existing_preview(self):
        self.capture.preview_region = (10, 20, 300, 400)
        self.capture.capture_click(self.event(50, 60))
        self.assertEqual(self.capture.region, (10, 20, 300, 400))
        self.capture.root.quit.assert_called_once()

    def test_manual_selection_normalizes_reverse_drag_and_clamps_to_desktop(self):
        self.capture.toggle_selection_mode()
        self.capture.capture_click(self.event(400, 300))
        self.capture.root.quit.assert_not_called()
        self.capture.finish_selection(self.event(-200, -100))
        self.assertEqual(self.capture.region, (-100, -50, 400, 300))
        self.capture.root.quit.assert_called_once()

    def test_zero_area_selection_can_be_retried(self):
        self.capture.toggle_selection_mode()
        self.capture.capture_click(self.event(20, 30))
        self.capture.finish_selection(self.event(20, 80))
        self.assertIsNone(self.capture.region)
        self.capture.root.quit.assert_not_called()
        self.capture.capture_click(self.event(20, 30))
        self.capture.finish_selection(self.event(70, 80))
        self.assertEqual(self.capture.region, (20, 30, 70, 80))

    def test_manual_preview_is_not_overwritten_by_timer(self):
        self.capture.toggle_selection_mode()
        self.capture.capture_click(self.event(20, 30))
        self.capture.drag_selection(self.event(70, 80))
        with patch.object(self.capture, 'window_at') as window_at:
            self.capture.update_preview()
        window_at.assert_not_called()
        self.assertEqual(self.capture.preview_region, (20, 30, 70, 80))
        self.capture.toggle_selection_mode()
        self.assertFalse(self.capture.manual_mode)
        self.assertIsNone(self.capture.drag_start)
        self.assertIsNone(self.capture.preview_region)


if __name__ == '__main__':
    unittest.main()
