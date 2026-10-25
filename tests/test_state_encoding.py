"""使用合成实体验证观测，不启动游戏窗口或加载模型。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from zaomeng_hanghai.core.entities import Bullet, Bullets, Enemy, EnemyList
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.observation import BULLET_SLICE, COLLISION_SLICE, ENEMY_SLICE, OBSERVATION_DIM
from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.env import Env


class StateEncodingTests(unittest.TestCase):
    def make_env(self, level=0, task=-1):
        env = Env.__new__(Env)
        env.player = SimpleNamespace(x=80, y=150, xn=0.25, yn=0.75)
        env.action_manager = SimpleNamespace(get_fire_cool_down=lambda: 0.3)
        # 刻意改变字典顺序，确保道具编码按名称读取。
        env.game_window = SimpleNamespace(
            hit_point=2, items={"fan": True, "dan": False, "pao": True}
        )
        env.map = SimpleNamespace(level=level, get_task=lambda: task, is_move_collision=Mock(return_value=False))
        env.enemy_list = EnemyList()
        env.bullets = Bullets()
        return env

    def test_start_area_normalization_and_empty_slots(self):
        state = self.make_env().state_encode()
        self.assertEqual(state.shape, (81,))
        self.assertEqual(OBSERVATION_DIM, 81)
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
        env.player = SimpleNamespace(x=0, y=0, xn=0, yn=0)
        for i in range(6, 0, -1):
            enemy = Enemy.__new__(Enemy)
            enemy.x, enemy.y = i * 32, 100
            enemy.xn, enemy.yn = i / 10, 0.5
            enemy.x1, enemy.y1 = enemy.x - 4, enemy.y - 3
            enemy.x2, enemy.y2 = enemy.x + 4, enemy.y + 3
            env.enemy_list.append(enemy)
            env.bullets.append(Bullet((i / 10, 0, i % 4)))
        env.bullets.update([], env.player)
        original_enemies = list(env.enemy_list)
        original_bullets = list(env.bullets)
        state = env.state_encode()
        enemies = state[ENEMY_SLICE].reshape(5, 7)
        bullets = state[BULLET_SLICE].reshape(3, 7)
        np.testing.assert_allclose(enemies[:, 1], [0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(bullets[:, 5], [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(enemies[:, 0], 1)
        np.testing.assert_array_equal(bullets[:, 0], 1)
        np.testing.assert_array_equal(bullets[:, 1:5], np.eye(4)[[1, 2, 3]])
        self.assertEqual(list(env.enemy_list), original_enemies)
        self.assertEqual(list(env.bullets), original_bullets)
        self.assertEqual(len(env.bullets), 6)  # 只限制观测槽位，不丢失追踪对象。

    def test_bullet_update_sorts_matched_missing_and_new_bullets(self):
        env = self.make_env()
        env.player = SimpleNamespace(x=0, y=0, xn=0, yn=0)
        env.bullets.extend([Bullet((0.8, 0, 2)), Bullet((0.6, 0, 2))])
        env.bullets.update([(0.79, 0, 2), (0.1, 0, 1)], env.player)
        np.testing.assert_allclose([b.xn for b in env.bullets], [0.1, 0.6, 0.79])
        self.assertEqual([b.disappear_count for b in env.bullets], [0, 1, 0])
        self.assertEqual(env.bullets.new_bullets, [env.bullets[0]])
        np.testing.assert_allclose(
            env.bullets.state_encode(env.player).reshape(3, 7)[:, 5], [0.1, 0.6, 0.79]
        )

    def test_partial_slots_are_zero_padded(self):
        env = self.make_env()
        enemy = Enemy.__new__(Enemy)
        enemy.x, enemy.y, enemy.xn, enemy.yn = 0, 0, 0, 0
        enemy.x1 = enemy.y1 = enemy.x2 = enemy.y2 = 0
        env.enemy_list.append(enemy)
        env.bullets.append(Bullet((0, 0, 0)))
        state = env.state_encode()
        np.testing.assert_array_equal(state[17:24], [1, -0.25, -0.75, -0.25, -0.75, -0.25, -0.75])
        np.testing.assert_array_equal(state[24:52], 0)
        np.testing.assert_array_equal(state[52:59], [1, 1, 0, 0, 0, -0.25, -0.75])
        np.testing.assert_array_equal(state[59:], 0)

    def test_actual_enemy_box_is_relative_to_player(self):
        env = self.make_env()
        enemy = Enemy(DetectionResult(1, [0.3, 0.4, 0.5, 0.6]), np.zeros((200, 320), dtype=bool))
        env.enemy_list.append(enemy)
        state = env.state_encode()
        np.testing.assert_allclose(state[18:24], [
            enemy.xn - 0.25, enemy.yn - 0.75,
            enemy.x1 / 320 - 0.25, enemy.y1 / 200 - 0.75,
            enemy.x2 / 320 - 0.25, enemy.y2 / 200 - 0.75,
        ])
        self.assertLess(state[21], 0)

    def test_collision_flags_have_exact_action_order_and_corner_rules(self):
        env = self.make_env(level=1, task=0)
        game_map = Map.__new__(Map)
        game_map.level = 1
        maze = np.zeros((200, 320), dtype=bool)
        maze[150, 81] = True
        maze[149, 80] = True
        game_map.level_mazes = [maze]
        game_map.map_template_without_exit = np.zeros_like(maze)
        game_map.get_task = lambda: 0
        env.map = game_map
        np.testing.assert_array_equal(env.state_encode()[COLLISION_SLICE], [1, 0, 0, 1, 1, 0, 1, 1])

    def test_entity_features_are_invariant_under_joint_translation(self):
        env = self.make_env()
        maze = np.zeros((200, 320), dtype=bool)
        env.enemy_list.append(Enemy(DetectionResult(1, [0.3, 0.4, 0.5, 0.6]), maze))
        env.bullets.append(Bullet((0.4, 0.6, 0)))
        before = env.state_encode()
        env.player = SimpleNamespace(x=120, y=175, xn=0.375, yn=0.875)
        env.enemy_list.clear()
        env.enemy_list.append(Enemy(DetectionResult(1, [0.425, 0.525, 0.625, 0.725]), maze))
        env.bullets.clear()
        env.bullets.append(Bullet((0.525, 0.725, 0)))
        np.testing.assert_allclose(env.state_encode()[17:73], before[17:73], atol=1e-7)

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
