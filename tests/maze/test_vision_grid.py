import unittest
from unittest.mock import patch

import cv2
import numpy as np

from zmxy4_treasure_hunt.maze.navigation.grid import Grid, GridDetectionError
from zmxy4_treasure_hunt.maze.vision import Character, DetectionError, detect_walls, locate_character, locate_game


class VisionTests(unittest.TestCase):
    def test_character_returns_center_and_positive_covering_radius_without_ui(self):
        image = np.full((120, 160, 3), 100, dtype=np.uint8)
        cv2.circle(image, (80, 60), 8, (0, 0, 0), -1)
        original = image.copy()
        with patch('cv2.imshow', side_effect=AssertionError('No UI in vision')):
            result = locate_character(image)
        self.assertEqual((result.x, result.y), (80, 60))
        self.assertGreaterEqual(result.radius, 15)
        np.testing.assert_array_equal(image, original)

    def test_bgr_wall_and_treasure_detection_keeps_source_unchanged(self):
        image = np.full((200, 200, 3), (150, 100, 50), dtype=np.uint8)
        for coordinate in (10, 20, 80, 90, 150, 160):
            cv2.line(image, (coordinate, 0), (coordinate, 199), (180, 180, 180), 5)
            cv2.line(image, (0, coordinate), (199, coordinate), (180, 180, 180), 5)
        cv2.rectangle(image, (110, 110), (124, 124), (40, 90, 180), -1)
        original = image.copy()
        observation = detect_walls(image, Character(50, 50, 8))
        self.assertTrue(observation.vertical_lines)
        self.assertTrue(observation.horizontal_lines)
        self.assertEqual(len(observation.treasures), 1)
        self.assertLessEqual(abs(observation.treasures[0][0] - 110), 2)
        self.assertLessEqual(abs(observation.treasures[0][1] - 110), 2)
        np.testing.assert_array_equal(image, original)

    def test_missing_character_and_lines_raise_clear_errors(self):
        with self.assertRaises(DetectionError):
            locate_character(np.full((100, 100, 3), 100, dtype=np.uint8))
        with self.assertRaises(DetectionError):
            detect_walls(np.zeros((100, 100, 3), dtype=np.uint8), Character(50, 50, 8))

    def test_gold_frame_and_corners_define_game_region(self):
        image = np.full((600, 1000, 3), 100, dtype=np.uint8)
        cv2.rectangle(image, (168, 138), (811, 505), (105, 180, 235), -1)
        cv2.rectangle(image, (180, 150), (799, 493), (10, 10, 10), -1)
        cv2.rectangle(image, (220, 190), (759, 453), (150, 200, 230), -1)
        first, last = locate_game(image)
        self.assertEqual(first, (180, 150))
        self.assertEqual(last, (800, 494))

    def test_missing_window_does_not_open_debug_windows(self):
        with patch('cv2.imshow', side_effect=AssertionError('No UI in vision')):
            with self.assertRaises(DetectionError):
                locate_game(np.zeros((100, 100, 3), dtype=np.uint8))


class GridTests(unittest.TestCase):
    def test_axes_are_calibrated_independently(self):
        grid = Grid(220, 400)
        grid.update([10, 20, 80, 90, 150, 160], [10, 30, 150, 170, 290, 310])
        self.assertEqual(grid.interval0x, 10)
        self.assertEqual(grid.interval0y, 20)
        self.assertEqual(grid.interval1x, 60)
        self.assertEqual(grid.interval1y, 120)

    def test_calibration_can_start_with_large_interval(self):
        grid = Grid(300, 300)
        grid.update([10, 70, 80, 140, 150], [10, 70, 80, 140, 150])
        self.assertEqual(grid.interval0x, 10)
        self.assertEqual(grid.interval1x, 60)

    def test_invalid_grid_does_not_poison_next_calibration(self):
        grid = Grid(220, 220)
        with self.assertRaises(GridDetectionError):
            grid.update([10, 20, 30], [10, 20, 30])
        self.assertEqual(grid.interval0x, 0)
        grid.update([10, 20, 80, 90], [10, 20, 80, 90])
        self.assertEqual(grid.interval1x, 60)

    def test_missing_lines_and_invalid_interval_fail_without_looping(self):
        grid = Grid(220, 220)
        for lines in ([], [10], [10, 11, 12], [0, 10, float('nan')]):
            with self.subTest(lines=lines), self.assertRaises(GridDetectionError):
                grid.update(lines, lines)
        for small, large in ((0, 1), (1, float('nan')), (-1, 5)):
            with self.subTest(small=small), self.assertRaises(GridDetectionError):
                grid.generate_expected_lines(10, 200, 0, small, large)

    def test_local_view_character_offsets_and_treasure(self):
        grid = Grid(160, 160)
        grid.vertical_lines = grid.horizontal_lines = [10, 20, 80, 90, 150]
        grid.interval1x = grid.interval1y = 60
        wall = np.zeros((160, 160), dtype=np.uint8)
        view, cell, offset = grid.analyze(wall, (50, 50), [(120, 120)])
        self.assertEqual(cell, (1, 1))
        self.assertEqual(offset, (0, 0))
        self.assertEqual(view[1, 2], 3)
        self.assertEqual(view[3, 3], 4)
        with self.assertRaises(GridDetectionError):
            grid.analyze(wall, (151, 50), [])
