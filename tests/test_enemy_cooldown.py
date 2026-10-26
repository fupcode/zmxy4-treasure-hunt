"""敌人射击冷却的更新与跨帧继承。"""

import unittest

import numpy as np

from zaomeng_hanghai.core.entities import Enemy, EnemyList
from zaomeng_hanghai.core.onnx_model import DetectionResult


class EnemyCooldownTests(unittest.TestCase):
    def setUp(self):
        self.maze = np.zeros((200, 320), dtype=bool)
        self.box = DetectionResult(1, [0.4, 0.4, 0.45, 0.5])

    def test_new_enemy_is_ready_without_fired_flag(self):
        enemy = Enemy(self.box, self.maze)
        self.assertFalse(hasattr(enemy, "fired"))
        self.assertEqual(enemy.fire_cool_down, 0)
        self.assertFalse(enemy.is_safe())

    def test_normal_enemy_countdown_expiry_and_reset(self):
        enemy = Enemy(self.box, self.maze)
        enemy.set_fired(100)
        self.assertEqual(enemy.fire_cool_down, 3)
        enemy.update_fire_cool_down(101.5)
        self.assertEqual(enemy.fire_cool_down, 1.5)
        enemy.update_fire_cool_down(102.9)
        self.assertTrue(enemy.is_safe())
        enemy.update_fire_cool_down(103)
        self.assertFalse(enemy.is_safe())
        enemy.update_fire_cool_down(110)
        self.assertEqual(enemy.fire_cool_down, 0)
        enemy.set_fired(111)
        self.assertEqual(enemy.fire_cool_down, 3)

    def test_boss_uses_one_second_cooldown(self):
        enemy = Enemy(DetectionResult(1, [0.4, 0.4, 0.5, 0.5]), self.maze)
        enemy.set_fired(100)
        self.assertEqual(enemy.fire_cool_down, 1)
        enemy.update_fire_cool_down(101)
        self.assertFalse(enemy.is_safe())

    def test_tracking_updates_matched_and_temporarily_missing_enemies(self):
        enemies = EnemyList()
        enemies.update([self.box], self.maze, current_time=100)
        enemies[0].set_fired(100)
        original = enemies[0]
        enemies.update([self.box], self.maze, current_time=101)
        self.assertIsNot(enemies[0], original)
        self.assertEqual(enemies[0].fire_cool_down, 2)
        enemies.update([], self.maze, current_time=102)
        self.assertEqual(enemies[0].fire_cool_down, 1)
        enemies.update([self.box], self.maze, current_time=104)
        self.assertEqual(enemies[0].fire_cool_down, 0)
        self.assertFalse(enemies[0].is_safe())

    def test_unmatched_enemy_does_not_inherit_old_cooldown(self):
        enemies = EnemyList()
        enemies.update([self.box], self.maze, current_time=100)
        enemies[0].set_fired(100)
        far_box = DetectionResult(1, [0.8, 0.4, 0.85, 0.5])
        enemies.update([far_box], self.maze, current_time=101)
        self.assertEqual(len(enemies), 2)
        self.assertEqual(enemies[0].fire_cool_down, 2)
        self.assertEqual(enemies[1].fire_cool_down, 0)


if __name__ == "__main__":
    unittest.main()
