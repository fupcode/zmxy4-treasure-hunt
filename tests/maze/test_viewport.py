from pathlib import Path
import unittest

import cv2
import numpy as np

from zmxy4_treasure_hunt.maze.viewport import locate_viewport
from zmxy4_treasure_hunt.maze.vision import DetectionError, locate_game

SOURCE = Path(__file__).resolve().parents[2] / "images" / "maze" / "full.png"


def draw_frame(image, box, color=(105, 180, 235)):
    left, top, right, bottom = box
    cv2.rectangle(image, (left - 12, top - 12), (right + 11, bottom + 11), color, -1)
    cv2.rectangle(image, (left, top), (right - 1, bottom - 1), (12, 12, 12), -1)
    cv2.rectangle(image, (left + 30, top + 30), (right - 31, bottom - 31), (150, 200, 230), -1)


class ViewportTests(unittest.TestCase):
    def test_exact_black_markers_without_frame_are_rejected(self):
        image = np.full((500, 900, 3), 100, np.uint8)
        cv2.rectangle(image, (40, 40), (50, 50), (1, 1, 1), -1)
        cv2.rectangle(image, (752, 440), (762, 450), (1, 1, 1), -1)
        with self.assertRaises(DetectionError):
            locate_game(image)

    def test_rectangular_dark_panel_needs_gold_on_all_four_sides(self):
        image = np.full((600, 1000, 3), 100, np.uint8)
        draw_frame(image, (180, 150, 800, 494), color=(170, 170, 170))
        with self.assertRaises(ValueError):
            locate_viewport(image)

    def test_wrong_aspect_ratio_is_rejected(self):
        image = np.full((600, 1000, 3), 100, np.uint8)
        draw_frame(image, (180, 150, 800, 450))
        with self.assertRaises(ValueError):
            locate_viewport(image)

    def test_missing_corner_is_rejected(self):
        image = np.full((600, 1000, 3), 100, np.uint8)
        draw_frame(image, (180, 150, 800, 494))
        cv2.rectangle(image, (165, 135), (220, 190), (100, 100, 100), -1)
        with self.assertRaises(ValueError):
            locate_viewport(image)

    def test_equally_good_separate_frames_are_ambiguous(self):
        image = np.full((1000, 1500, 3), 100, np.uint8)
        draw_frame(image, (50, 80, 670, 424))
        draw_frame(image, (800, 500, 1420, 844))
        with self.assertRaisesRegex(ValueError, '多个'):
            locate_viewport(image)

    def test_blackout_inside_frame_still_allows_window_location(self):
        image = np.full((600, 1000, 3), 100, np.uint8)
        box = (180, 150, 800, 494)
        draw_frame(image, box)
        image[150:494, 180:800] = 0
        self.assertEqual(locate_viewport(image).box, box)


@unittest.skipUnless(SOURCE.exists(), '缺少 full.png 实际截图')
class RealViewportTests(unittest.TestCase):
    def setUp(self):
        self.image = cv2.imread(str(SOURCE))
        self.expected = np.array((461, 374, 1272, 824))

    def assert_box_close(self, image, expected, tolerance=3):
        match = locate_viewport(image)
        self.assertLessEqual(np.max(np.abs(np.asarray(match.box) - expected)), tolerance)
        self.assertGreaterEqual(match.confidence, 0.8)
        self.assertGreaterEqual(min(match.gold_coverage), 0.6)
        self.assertGreaterEqual(min(match.corner_support), 0.35)

    def test_full_screenshot_and_scaled_variants(self):
        for scale in (0.5, 0.75, 1, 1.25, 1.5):
            with self.subTest(scale=scale):
                image = cv2.resize(self.image, None, fx=scale, fy=scale)
                self.assert_box_close(image, self.expected * scale)

    def test_brightness_variants(self):
        for brightness in (0.7, 1.2):
            with self.subTest(brightness=brightness):
                image = np.clip(self.image.astype(float) * brightness, 0, 255).astype(np.uint8)
                self.assert_box_close(image, self.expected)

    def test_jpeg_compression(self):
        ok, encoded = cv2.imencode('.jpg', self.image, [cv2.IMWRITE_JPEG_QUALITY, 60])
        self.assertTrue(ok)
        self.assert_box_close(cv2.imdecode(encoded, cv2.IMREAD_COLOR), self.expected)

    def test_translation_and_screenshot_are_not_modified(self):
        original = self.image.copy()
        translated = cv2.copyMakeBorder(self.image, 35, 20, 60, 10, cv2.BORDER_CONSTANT, value=(100, 100, 100))
        self.assert_box_close(translated, self.expected + (60, 35, 60, 35))
        np.testing.assert_array_equal(self.image, original)
