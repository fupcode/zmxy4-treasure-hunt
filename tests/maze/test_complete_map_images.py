"""有本地原图时验证实际墙线识别到完整地图定位的流程。"""

from pathlib import Path
import unittest

import cv2
import numpy as np

from zmxy4_treasure_hunt.maze.navigation.templates import load_weekday_map
from zmxy4_treasure_hunt.maze.navigation import Maze
from zmxy4_treasure_hunt.maze.vision import Character, detect_walls

IMAGES = Path(__file__).resolve().parents[2] / "images" / "maze"


class CompleteMapImageTests(unittest.TestCase):
    @unittest.skipUnless(all((IMAGES / f"ft{i}.png").exists() for i in range(7)), "缺少本地迷宫原图")
    def test_seven_source_images_align_with_weekly_maps(self):
        for weekday in range(7):
            with self.subTest(weekday=weekday):
                image = cv2.imread(str(IMAGES / f"ft{weekday}.png"))
                # 原图没有人物，提供左上房间内的坐标，单独验证墙线/格网/定位链路。
                character = Character(48, 44, 12)
                observation = detect_walls(image, character)
                template = load_weekday_map(weekday)
                maze = Maze(image.shape[1], image.shape[0], template.cells)
                result = maze.analyze(observation, character)
                self.assertEqual(maze.char_coord, (1, 1))
                self.assertEqual(maze.maze[result.path[-1]], 4)
                self.assertFalse(result.complete)
                np.testing.assert_array_equal(maze.maze, template.cells)
