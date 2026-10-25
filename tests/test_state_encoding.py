"""使用合成实体验证观测，不启动游戏窗口或加载模型。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from zaomeng_hanghai.core.entities import Bullet, Bullets, Enemy, EnemyList
from zaomeng_hanghai.env import Env


class StateEncodingTests(unittest.TestCase):
    def make_env(self, level=0, task=-1):
        env = Env.__new__(Env)
        env.player = SimpleNamespace(x=0, y=0, xn=0.25, yn=0.75)
        env.action_manager = SimpleNamespace(get_fire_cool_down=lambda: 0.3)
        # 刻意改变字典顺序，确保道具编码按名称读取。
        env.game_window = SimpleNamespace(
            hit_point=2, items={"fan": True, "dan": False, "pao": True}
        )
        env.map = SimpleNamespace(level=level, get_task=lambda: task)
        env.enemy_list = EnemyList()
        env.bullets = Bullets()
        return env

    def test_start_area_normalization_and_empty_slots(self):
        state = self.make_env().state_encode()
        self.assertEqual(state.shape, (60,))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state[:8], [0.25, 0.75, 0.5, 0.5, 1, 0, 1, 0])
        np.testing.assert_array_equal(state[8:], 0)

    def test_all_task_types_and_level_range(self):
        for task in range(9):
            state = self.make_env(level=4, task=task).state_encode()
            self.assertEqual(state[7], 1)
            self.assertEqual(state[8:17].sum(), 1)
            self.assertEqual(state[8 + task], 1)
        with self.assertRaises(ValueError):
            self.make_env(task=9).state_encode()

    def test_nearest_entities_truncation_and_layout(self):
        env = self.make_env()
        for i in range(6, 0, -1):
            enemy = Enemy.__new__(Enemy)
            enemy.x, enemy.y = i * 10, 0
            enemy.xn, enemy.yn = i / 10, 0.5
            env.enemy_list.append(enemy)
            env.bullets.append(Bullet((i / 10, 0, i % 4)))
        env.bullets.update([], env.player)
        original_enemies = list(env.enemy_list)
        original_bullets = list(env.bullets)
        state = env.state_encode()
        enemies = state[17:32].reshape(5, 3)
        bullets = state[32:].reshape(4, 7)
        np.testing.assert_allclose(enemies[:, 1], [0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(bullets[:, 5], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(enemies[:, 0], 1)
        np.testing.assert_array_equal(bullets[:, 0], 1)
        np.testing.assert_array_equal(bullets[:, 1:5], np.eye(4)[[1, 2, 3, 0]])
        self.assertEqual(list(env.enemy_list), original_enemies)
        self.assertEqual(list(env.bullets), original_bullets)

    def test_bullet_update_sorts_matched_missing_and_new_bullets(self):
        env = self.make_env()
        env.bullets.extend([Bullet((0.8, 0, 2)), Bullet((0.6, 0, 2))])
        env.bullets.update([(0.79, 0, 2), (0.1, 0, 1)], env.player)
        np.testing.assert_allclose([b.xn for b in env.bullets], [0.1, 0.6, 0.79])
        self.assertEqual([b.disappear_count for b in env.bullets], [0, 1, 0])
        self.assertEqual(env.bullets.new_bullets, [env.bullets[0]])
        np.testing.assert_allclose(
            env.bullets.state_encode().reshape(4, 7)[:, 5], [0.1, 0.6, 0.79, 0]
        )

    def test_partial_slots_are_zero_padded(self):
        env = self.make_env()
        enemy = Enemy.__new__(Enemy)
        enemy.x, enemy.y, enemy.xn, enemy.yn = 0, 0, 0, 0
        env.enemy_list.append(enemy)
        env.bullets.append(Bullet((0, 0, 0)))
        state = env.state_encode()
        np.testing.assert_array_equal(state[17:20], [1, 0, 0])
        np.testing.assert_array_equal(state[20:32], 0)
        np.testing.assert_array_equal(state[32:39], [1, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(state[39:], 0)

    def test_terminal_update_skips_scene_processing(self):
        for status in (1, -2):
            env = self.make_env(level=4, task=8)
            env._reset_combat_timer()
            env.game_window.update = Mock(return_value=status)
            env.map.get_maze = Mock(side_effect=AssertionError("终局不应更新地图"))
            self.assertEqual(env.update(), status)
            env.map.get_maze.assert_not_called()
            self.assertEqual(env.state_encode()[7], 1)


if __name__ == "__main__":
    unittest.main()
