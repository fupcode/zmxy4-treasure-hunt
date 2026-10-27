"""射击目标与实际开火共用对齐、射程和遮挡规则。"""

from types import SimpleNamespace
import unittest

import numpy as np

from zmxy4_treasure_hunt.voyage.core.constants import FIRE_X_RANGE, FIRE_Y_RANGE, MAZE_HEIGHT, MAZE_WIDTH, VISION
from zmxy4_treasure_hunt.voyage.core.entities import Enemy
from zmxy4_treasure_hunt.voyage.core.map import Map


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
        enemy.distance_array = enemy.fire_range = None
        enemy.fire_cool_down = cooldown
        enemy.cool_down_time = 3
        return enemy

    def regions(self, enemies, x=30, y=30):
        self.game_map.flee_mazes = self.game_map.get_flee_maze(
            SimpleNamespace(x=x, y=y), [], enemies,
        )
        return self.game_map.flee_mazes

    def test_flee_radius_scales_with_each_enemys_cooldown_progress(self):
        for level, base_radius, duration in ((1, 40, 3), (4, 56, 1)):
            self.game_map.level = level
            enemy = self.enemy()
            enemy.cool_down_time = duration
            for progress in (0, 0.375, 0.5, 1):
                with self.subTest(level=level, progress=progress):
                    enemy.fire_cool_down = duration * (1 - progress)
                    danger = self.regions([enemy])[3]
                    expected = np.maximum(0, base_radius * progress - enemy.distance_array)
                    np.testing.assert_allclose(danger, expected)

        self.game_map.level = 1
        ready = self.enemy(x=80, y=80)
        cooling = self.enemy(x=220, y=120, cooldown=3)
        np.testing.assert_array_equal(self.regions([ready, cooling])[3], self.regions([ready])[3])


    def test_lower_alignment_edges_cannot_be_targets_or_stop_positions(self):
        enemy = self.enemy()
        for x, y in ((150, 40), (80, 100)):
            with self.subTest(x=x, y=y):
                regions = self.regions([enemy], x, y)
                self.assertEqual((regions[-1] + regions[-2])[y, x], 0)
                player = SimpleNamespace(x=x, y=y)
                self.assertIsNone(self.game_map.fire_direction(player, [enemy]))
                direction, path, mode = self.game_map.get_shot_place_direction(player)
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
        self.assertGreater(enemy.fire_range[5, 160], 0)
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
            targets = regions[-1] + regions[-2]
            self.assertTrue((targets > 0).any())
            for y, x in np.argwhere(targets > 0):
                self.assertIsNotNone(self.game_map.fire_direction(SimpleNamespace(x=x, y=y), enemies))
            self.assertTrue(np.all(targets[regions[2] + regions[3] > 0] == 0))
            if level == 4:
                for enemy in enemies:
                    if not enemy.is_safe():
                        self.assertFalse(targets[:enemy.y2, enemy.x1:enemy.x2].any())

    def test_vision_square_includes_corners_and_ignores_walls(self):
        enemy = self.enemy()
        self.maze[:, 140] = True
        regions = self.regions([enemy])
        vision = regions[4]
        self.assertEqual(vision[enemy.y - VISION, enemy.x - VISION], 1)
        self.assertEqual(vision[enemy.y + VISION, enemy.x + VISION], 1)
        self.assertEqual(vision[enemy.y, enemy.x - VISION - 1], 0)
        self.assertEqual(vision[enemy.y, 140], 1)
        self.assertFalse(regions[-1][vision > 0].any())
        self.assertFalse(regions[-2][vision == 0].any())
        self.assertEqual(regions[5][enemy.y - VISION, enemy.x], 1)
        self.assertEqual(regions[5][enemy.y - VISION, enemy.x - VISION], 0)
        clipped = self.enemy(x=2, y=3, box=(0, 0, 5, 6)).get_vision_region(self.maze.shape)
        self.assertTrue(clipped[0, 0])
        self.assertFalse(clipped[-1, -1])

    def test_cooldown_removes_vision_for_normal_and_boss(self):
        enemy = self.enemy()
        for level in (1, 4):
            self.game_map.level = level
            ready = self.regions([enemy])
            self.assertTrue(ready[4].any())
            if level == 4:
                self.assertTrue(ready[4].all())
                self.assertFalse(ready[-1].any())
            enemy.fire_cool_down = 1
            cooling = self.regions([enemy])
            self.assertFalse(cooling[4].any())
            self.assertFalse(cooling[5].any())
            self.assertFalse(cooling[-2].any())
            self.assertTrue(cooling[-1].any())
            enemy.fire_cool_down = 0
            self.assertTrue(self.regions([enemy])[4].any())

    def test_cross_counts_overlap_and_keeps_same_enemy_for_local_targets(self):
        first = self.enemy(x=100, y=100, box=(95, 95, 105, 105))
        second = self.enemy(x=170, y=100, box=(165, 95, 175, 105), cooldown=1)
        # x=170 在 first 视野外；second 冷却时不提供局部对齐范围。
        regions = self.regions([first, second])
        self.assertEqual(regions[5][50, 170], 0)
        self.assertEqual(regions[-2][50, 170], 0)
        second.fire_cool_down = 0
        regions = self.regions([first, second])
        self.assertEqual(regions[4][100, 135], 2)
        self.assertEqual(regions[5][100, 135], 2)
        self.assertTrue(np.all(regions[-1][regions[4] > 0] == 0))
        # 在 first 的视野内只能命中冷却中的 second，不能借用 first 的视野做局部终点。
        second = self.enemy(x=220, y=100, box=(215, 95, 225, 105), cooldown=1)
        self.maze[100, 130] = True
        regions = self.regions([first, second])
        player = SimpleNamespace(x=160, y=100)
        self.assertIsNotNone(self.game_map.fire_direction(player, [second]))
        self.assertEqual(regions[4][100, 160], 1)
        self.assertEqual(regions[5][100, 160], 1)
        self.assertEqual(regions[-2][100, 160], 0)
        self.assertEqual(regions[-1][100, 160], 0)


if __name__ == "__main__":
    unittest.main()
