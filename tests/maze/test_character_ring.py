from pathlib import Path
import unittest

import cv2
import numpy as np

from zmxy4_treasure_hunt.maze.vision.mask import MaskRestorer
from zmxy4_treasure_hunt.maze.vision import DetectionError, locate_character

IMAGES = Path(__file__).resolve().parents[2] / "images" / "maze"


def partial_ring(scale=1, angle=0, hidden=0):
    image = np.full((344, 620, 3), (200, 220, 240), np.uint8)
    cv2.circle(image, (310, 172), 20, (150, 100, 60), -1)
    cv2.ellipse(image, (310, 172), (24, 24), angle, hidden / 2, 360 - hidden / 2, (12, 12, 12), 4)
    return cv2.resize(image, None, fx=scale, fy=scale)


class CharacterRingTests(unittest.TestCase):
    def test_partial_ring_across_scales_and_occlusion_directions(self):
        for scale in (0.5, 1, 1.5):
            for angle in (0, 90, 180, 270):
                for hidden in (90, 180):
                    with self.subTest(scale=scale, angle=angle, hidden=hidden):
                        character = locate_character(partial_ring(scale, angle, hidden))
                        self.assertLessEqual(np.hypot(character.x / scale - 310, character.y / scale - 172), 5)
                        self.assertIsNotNone(character.ring_radius)
                        self.assertLessEqual(abs(character.ring_radius / scale - 24), 4)

    def test_ring_does_not_require_exact_black(self):
        image = partial_ring()
        image = np.clip(image.astype(int) + 5, 0, 255).astype(np.uint8)
        before = image.copy()
        character = locate_character(image)
        self.assertLessEqual(np.hypot(character.x - 310, character.y - 172), 3)
        np.testing.assert_array_equal(image, before)

    def test_straight_walls_rectangle_and_small_dot_do_not_identify_character(self):
        for kind in ('walls', 'rectangle', 'dot'):
            image = np.full((344, 620, 3), 220, np.uint8)
            if kind == 'walls':
                cv2.line(image, (100, 50), (100, 300), (10, 10, 10), 5)
                cv2.line(image, (100, 200), (500, 200), (10, 10, 10), 5)
            elif kind == 'rectangle':
                cv2.rectangle(image, (270, 130), (350, 210), (10, 10, 10), 4)
            else:
                cv2.circle(image, (310, 172), 2, (0, 0, 0), -1)
            with self.subTest(kind=kind), self.assertRaises(DetectionError):
                locate_character(image)

    def test_ring_in_untrusted_region_is_ignored(self):
        image = partial_ring()
        with self.assertRaises(DetectionError):
            locate_character(image, np.zeros(image.shape[:2], bool))

    def test_two_equally_plausible_rings_are_rejected(self):
        image = np.full((344, 620, 3), 220, np.uint8)
        for x in (150, 450):
            cv2.circle(image, (x, 172), 24, (10, 10, 10), 4)
        with self.assertRaises(DetectionError):
            locate_character(image)


class RealCharacterRingTests(unittest.TestCase):
    @unittest.skipUnless(all((IMAGES / f'ch{i}.png').exists() for i in (2, 5, 6, 7)), '缺少四个人物截图')
    def test_four_characters_at_different_resolutions(self):
        centers = {2: (154, 143), 5: (155, 143), 6: (143, 148), 7: (146, 146)}
        for number, expected in centers.items():
            original = cv2.imread(str(IMAGES / f'ch{number}.png'))
            for scale in (0.25, 0.5, 0.625, 1, 1.5, 2):
                with self.subTest(character=number, scale=scale):
                    image = cv2.resize(original, None, fx=scale, fy=scale)
                    frame = MaskRestorer(image.shape[1], image.shape[0]).restore(image)
                    character = locate_character(frame.image, frame.valid_pixels)
                    sx, sy = image.shape[1] / original.shape[1], image.shape[0] / original.shape[0]
                    x, y = (character.x + 0.5) / sx - 0.5, (character.y + 0.5) / sy - 0.5
                    self.assertLessEqual(np.hypot(x - expected[0], y - expected[1]), 8)

    @unittest.skipUnless(all((IMAGES / f'ft{i}.png').exists() for i in range(7)), '缺少无人物完整地图')
    def test_treasures_and_map_walls_are_not_characters(self):
        for day in range(7):
            with self.subTest(day=day), self.assertRaises(DetectionError):
                locate_character(cv2.imread(str(IMAGES / f'ft{day}.png')))
