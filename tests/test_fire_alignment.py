"""射击目标与实际开火共用对齐、射程和遮挡规则。"""

from types import SimpleNamespace
import unittest

import numpy as np

from zaomeng_hanghai.core.constants import FIRE_X_RANGE, FIRE_Y_RANGE, MAZE_HEIGHT, MAZE_WIDTH
from zaomeng_hanghai.core.entities import Enemy
from zaomeng_hanghai.core.map import Map


class FireAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.maze = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
        self.game_map = Map.__new__(Map)
        self.game_map.level = 1
        self.game_map.render_enabled = False
        self.game_map.get_maze = lambda: self.maze
        self.game_map.get_distance_maze = lambda: np.ones(self.maze.shape)
        self.game_map.map_template_without_exit = np.zeros_like(self.maze)

    def enemy(self, x=160, y=105, box=(150, 100, 170, 110), cooldown=0):
        enemy = Enemy.__new__(Enemy)
        enemy.x, enemy.y = x, y
        enemy.x1, enemy.y1, enemy.x2, enemy.y2 = box
        enemy.distance_array = enemy.vision = None
        enemy.fire_cool_down = cooldown
        return enemy

    def regions(self, enemies, x=30, y=30):
        self.game_map.flee_mazes = self.game_map.get_flee_maze(
            SimpleNamespace(x=x, y=y), [], enemies,
        )
        return self.game_map.flee_mazes

    def test_lower_alignment_edges_cannot_be_targets_or_stop_positions(self):
        enemy = self.enemy()
        for x, y in ((150, 40), (80, 100)):
            with self.subTest(x=x, y=y):
                regions = self.regions([enemy], x, y)
                self.assertEqual(regions[-1][y, x], 0)
                player = SimpleNamespace(x=x, y=y)
                self.assertIsNone(self.game_map.fire_direction(player, [enemy]))
                direction, path, mode = self.game_map.get_shot_place_direction(player, None, None)
                self.assertNotEqual(mode, 0)
                self.assertIsNotNone(direction)
                self.assertTrue(path)
                destination = SimpleNamespace(x=path[-1][0], y=path[-1][1])
                self.assertIsNotNone(self.game_map.fire_direction(destination, [enemy]))

    def test_range_uses_integer_center_and_includes_exact_limit(self):
        enemy = self.enemy()
        for x, y, expected in (
            (160, 105 - FIRE_Y_RANGE, True),
            (160, 105 - FIRE_Y_RANGE - 1, False),
            (160 - FIRE_X_RANGE, 105, True),
            (160 - FIRE_X_RANGE - 1, 105, False),
            (160 + FIRE_X_RANGE, 105, True),
            (160 + FIRE_X_RANGE + 1, 105, False),
        ):
            with self.subTest(x=x, y=y):
                region = self.game_map.get_enemy_fire_region(enemy)
                self.assertEqual(bool(region[y, x]), expected)
                distance, _ = self.game_map.fire_distance(SimpleNamespace(x=x, y=y), enemy)
                self.assertEqual(distance is not None, expected)

    def test_large_box_does_not_extend_effective_fire_range(self):
        enemy = self.enemy(y=115, box=(150, 100, 170, 130))
        region = self.regions([enemy], 160, 5)[-1]
        self.assertGreater(enemy.vision[5, 160], 0)
        self.assertEqual(region[5, 160], 0)
        self.assertIsNone(self.game_map.fire_direction(SimpleNamespace(x=160, y=5), [enemy]))

    def test_obstacles_block_exact_ray_including_endpoints(self):
        enemy = self.enemy()
        self.maze[60, 160] = True
        region = self.game_map.get_enemy_fire_region(enemy)
        self.assertFalse(region[40, 160])
        self.assertFalse(region[60, 160])
        self.assertTrue(region[61, 160])
        self.assertTrue(region[40, 161])  # 邻接射线不受 5 格条带牵连。
        for y in (40, 60, 61):
            actual = self.game_map.fire_distance(SimpleNamespace(x=160, y=y), enemy)[0]
            self.assertEqual(actual is not None, bool(region[y, 160]))

    def test_all_valid_targets_can_fire_for_multiple_enemies_and_cooldown(self):
        self.maze[60, 160] = True
        self.maze[105, 100] = True
        enemies = [self.enemy(), self.enemy(240, 150, (230, 145, 250, 155), cooldown=1)]
        for level in (1, 4):
            self.game_map.level = level
            regions = self.regions(enemies)
            targets = regions[-1]
            self.assertTrue((targets > 0).any())
            for y, x in np.argwhere(targets > 0):
                self.assertIsNotNone(self.game_map.fire_direction(SimpleNamespace(x=x, y=y), enemies))
            self.assertTrue(np.all(targets[regions[2] + regions[3] > 0] == 0))
            if level == 4:
                for enemy in enemies:
                    if not enemy.is_safe():
                        self.assertFalse(targets[:enemy.y2, enemy.x1:enemy.x2].any())


if __name__ == "__main__":
    unittest.main()
