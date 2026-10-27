from pathlib import Path
import unittest

import cv2
import numpy as np

from zmxy4_treasure_hunt.maze.navigation.grid import Grid
from zmxy4_treasure_hunt.maze.navigation.templates import load_weekday_map
from zmxy4_treasure_hunt.maze.vision.mask import MaskRestorer
from zmxy4_treasure_hunt.maze.navigation import Maze
from zmxy4_treasure_hunt.maze.vision import detect_walls, locate_character, locate_game

SOURCE = Path(__file__).resolve().parents[2] / "images" / "maze" / "full.png"


class MaskTests(unittest.TestCase):
    def test_known_alpha_recovers_bgr_with_quantization_error(self):
        alpha = np.array([[0, 64, 128, 192]], dtype=np.uint8)
        original = np.full((1, 4, 3), (40, 120, 220), dtype=np.uint8)
        transmission = 1 - alpha.astype(float) / 255
        observed = np.rint(original * transmission[..., None]).astype(np.uint8)
        copy = observed.copy()
        frame = MaskRestorer(4, 1, alpha=alpha).restore(observed)
        self.assertLessEqual(np.abs(frame.image.astype(int) - original).max(), 2)
        np.testing.assert_array_equal(frame.image[0, 0], original[0, 0])
        np.testing.assert_array_equal(observed, copy)

    def test_opaque_pixels_are_preserved_and_low_transmission_is_not_trusted(self):
        alpha = np.array([[0, 230, 253, 255]], dtype=np.uint8)
        image = np.full((1, 4, 3), 10, dtype=np.uint8)
        restorer = MaskRestorer(4, 1, alpha=alpha)
        frame = restorer.restore(image)
        self.assertGreater(frame.image[0, 1, 0], image[0, 1, 0])
        np.testing.assert_array_equal(frame.valid_pixels, [[True, False, False, False]])
        np.testing.assert_array_equal(frame.image[:, 2:], image[:, 2:])
        self.assertFalse(frame.valid_pixels.flags.writeable)
        self.assertIs(restorer.restore(image).valid_pixels, frame.valid_pixels)

    def test_invalid_parameters_and_image_size_are_rejected(self):
        with self.assertRaises(ValueError):
            MaskRestorer(5, 5, min_transmission=0)
        with self.assertRaises(ValueError):
            MaskRestorer(5, 5, min_transmission=0.2, recognition_transmission=0.1)
        with self.assertRaises(ValueError):
            MaskRestorer(5, 5, alpha=np.zeros((3, 3), dtype=float))
        restorer = MaskRestorer(5, 5, alpha=np.zeros((3, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            restorer.restore(np.zeros((6, 5, 3), dtype=np.uint8))

    def test_untrusted_black_border_cannot_be_mistaken_for_character(self):
        image = np.full((240, 240, 3), 100, np.uint8)
        cv2.rectangle(image, (0, 0), (239, 239), (0, 0, 0), 70)
        cv2.circle(image, (120, 120), 8, (0, 0, 0), -1)
        valid = np.zeros((240, 240), dtype=bool)
        valid[40:200, 40:200] = True
        character = locate_character(image, valid)
        self.assertEqual((character.x, character.y), (120, 120))
        self.assertLess(character.radius, 20)

    def test_untrusted_connections_are_unknown_in_frame_not_false_walls(self):
        grid = Grid(160, 160)
        grid.vertical_lines = grid.horizontal_lines = [10, 20, 80, 90, 150]
        grid.interval1x = grid.interval1y = 60
        wall = np.full((160, 160), 255, np.uint8)
        valid = np.ones((160, 160), dtype=bool)
        valid[20:80, 80:90] = False
        view, _, _ = grid.analyze(wall, (50, 50), [], valid)
        self.assertEqual(view[1, 2], 0)
        self.assertEqual(view[2, 1], 2)


@unittest.skipUnless(SOURCE.exists(), '缺少 full.png 实际截图')
class RealMaskTests(unittest.TestCase):
    def test_full_screenshot_restoration_character_and_complete_map_location(self):
        image = cv2.imread(str(SOURCE))
        (left, top), (right, bottom) = locate_game(image)
        raw = image[top:bottom, left:right]
        original = raw.copy()
        frame = MaskRestorer(right - left, bottom - top).restore(raw)
        self.assertGreater(frame.image.mean(), raw.mean())
        self.assertGreater(frame.valid_pixels.mean(), 0.7)
        character = locate_character(frame.image, frame.valid_pixels)
        self.assertLessEqual(abs(character.x - 116), 3)
        self.assertLessEqual(abs(character.y - 116), 3)
        walls = detect_walls(frame.image, character, frame.valid_pixels)
        self.assertTrue(np.all(walls.image[~frame.valid_pixels] == 0))
        self.assertIs(walls.valid_pixels, frame.valid_pixels)
        # 本截图左上区域与 2.json 对应；不依赖测试当天的星期。
        template = load_weekday_map(2)
        maze = Maze(right - left, bottom - top, template.cells)
        result = maze.analyze(walls, character)
        self.assertEqual(maze.char_coord, (1, 1))
        self.assertEqual(maze.maze[result.path[-1]], 4)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(maze.maze, template.cells)
