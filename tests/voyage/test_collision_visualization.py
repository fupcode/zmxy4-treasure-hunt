"""验证障碍框不会覆盖空地，以及所有实体类型都能被可视化。"""

from types import SimpleNamespace
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tests.collision_test import draw_scene, mask_rectangles, raw_obstacle_mask
from zmxy4_treasure_hunt.voyage.core.entities import Bullets, EnemyList, FallingItems, Player
from zmxy4_treasure_hunt.voyage.core.map import Map
from zmxy4_treasure_hunt.voyage.core.onnx_model import DetectionResult


class CollisionVisualizationTests(unittest.TestCase):
    def test_rectangles_cover_exact_mask_including_holes_and_edges(self):
        mask = np.array([
            [1, 1, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 0, 0], [0, 0, 1, 0, 1]
        ], dtype=bool)
        reconstructed = np.zeros_like(mask)
        for x1, y1, x2, y2 in mask_rectangles(mask):
            self.assertFalse(reconstructed[y1:y2, x1:x2].any())
            reconstructed[y1:y2, x1:x2] = True
        np.testing.assert_array_equal(reconstructed, mask)
        self.assertEqual(mask_rectangles(np.zeros((3, 5), dtype=bool)), [])

    def test_all_entity_layers_and_map_are_drawn(self):
        game_map = Map(render_enabled=False)
        game_map.reset(0)
        game_map.next_level()
        maze = game_map.get_maze()
        raw = raw_obstacle_mask(game_map)
        self.assertEqual(raw.shape, (200, 320))
        self.assertTrue(raw.any())
        self.assertTrue(maze[raw].all())
        player_box = DetectionResult(0.9, [0.4, 0.4, 0.45, 0.45])
        enemy_box = DetectionResult(0.9, [0.6, 0.4, 0.65, 0.45])
        player = Player(player_box, maze)
        enemies, bullets, items = EnemyList(), Bullets(), FallingItems()
        enemies.update([enemy_box], maze)
        bullets.update([(0.3, 0.42, 0), (0.8, 0.42, 0)], player)
        items.update([(0.5, 0.5)])
        window = SimpleNamespace(detections={"player": [player_box], "enemy": [enemy_box]})
        fig = draw_scene(np.zeros((200, 320, 3), dtype=np.uint8), window, game_map,
                         player, enemies, bullets, items, "synthetic")
        try:
            labels = [text.get_text() for ax in fig.axes for text in ax.texts]
            for name in ("P", "E1", "B1", "S1", "I1", "O1", "EXIT 0"):
                self.assertIn(name, labels)
            self.assertEqual(len(fig.axes), 2)
            fig.canvas.draw()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
