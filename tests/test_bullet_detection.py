"""仅凭 YOLO 框与弹头颜色定位、定向，不要求火焰轮廓。"""

import unittest

import numpy as np

from zmxy4_treasure_hunt.voyage.core.onnx_model import DetectionResult
from zmxy4_treasure_hunt.voyage.game_window import GameWindow


class BulletDetectionTests(unittest.TestCase):
    def setUp(self):
        self.window = GameWindow.__new__(GameWindow)
        self.boxes = [DetectionResult(0.9, [0.2, 0.2, 0.8, 0.8])]

    def test_four_directions_without_any_flame_pixels(self):
        for direction, (x, y) in enumerate(((70, 50), (50, 70), (30, 50), (50, 30))):
            with self.subTest(direction=direction):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                image[y - 1:y + 2, x - 1:x + 2] = (20, 100, 20)
                source = image.copy()
                bullets = self.window.detect_bullet_normal(image, self.boxes)
                self.assertEqual(len(bullets), 1)
                xn, yn, actual_direction = bullets[0]
                self.assertEqual(actual_direction, direction)
                self.assertAlmostEqual(xn, (x + 0.5) / 100)
                self.assertAlmostEqual(yn, (y + 0.5) / 100)
                np.testing.assert_array_equal(image, source)

    def test_largest_head_wins_over_noise_and_outside_pixels(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[49:52, 69:72] = (20, 100, 20)
        image[25, 25] = (20, 100, 20)
        image[:10, :10] = (20, 100, 20)
        bullets = self.window.detect_bullet_normal(image, self.boxes)
        self.assertEqual(bullets, [(0.705, 0.505, 0)])

    def test_no_head_or_centered_head_does_not_invent_direction(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertEqual(self.window.detect_bullet_normal(image, self.boxes), [])
        image[49:51, 49:51] = (20, 100, 20)
        self.assertEqual(self.window.detect_bullet_normal(image, self.boxes), [])
        self.assertEqual(self.window.detect_bullet_normal(image, []), [])


if __name__ == '__main__':
    unittest.main()
