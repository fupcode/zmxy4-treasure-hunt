"""验证感知更新、自动战斗寻路和过关接管的计算边界。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from zaomeng_hanghai.core.entities import Bullets, Enemy, EnemyList, FallingItems
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.core.rewards import RewardConfig
from zaomeng_hanghai.env import Env


class NextLevelTests(unittest.TestCase):
    def test_next_level_has_no_extra_sleep_and_preserves_status(self):
        for status in (0, 1, -2):
            with self.subTest(status=status):
                env = Env.__new__(Env)
                env.combat_timeout = 120
                env._reset_combat_timer()
                env.new_level = False
                env.stay_timeout = 30
                env.player = Mock()
                env.falling_items = []
                env.map = Mock()
                env.map.get_next_direction.return_value = (None, [])
                env.action_manager = Mock()
                env.render = Mock()
                env.update = Mock(return_value=status)
                env.can_next_level = Mock(side_effect=[True, False])
                with patch("zaomeng_hanghai.env.time.sleep") as sleep:
                    self.assertEqual(env.try_next_level(), status)
                sleep.assert_not_called()
                env.action_manager.take_action_direction.assert_called_once_with(None)
                env.update.assert_called_once_with()

    def test_observation_update_never_builds_danger_map(self):
        for enemies, bullets in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(enemies=enemies, bullets=bullets):
                env = Env.__new__(Env)
                env.combat_timeout = 120
                env._reset_combat_timer()
                env.player = None
                env.render_enabled = False
                env.fps_calculator = Mock()
                env.game_window = Mock()
                env.game_window.update.return_value = 0
                env.game_window.items = {"pao": False}
                env.map = Mock(level=1)
                env.action_manager = Mock()
                env.enemy_list = MagicMock()
                env.enemy_list.__len__.return_value = enemies
                env.bullets = MagicMock()
                env.bullets.__len__.return_value = bullets
                env.bullets.new_bullets = []
                env.falling_items = Mock()
                with patch("zaomeng_hanghai.env.Player"):
                    self.assertEqual(env.update(), 0)
                env.fps_calculator.trigger.assert_called_once_with(log_to_terminal=True)
                env.map.update.assert_not_called()
                self.assertIsNone(env.map.debug_image)
                self.assertIsNone(env.map.flee_mazes)

    def test_agent_step_updates_encoded_state_and_keeps_next_level_handler(self):
        env = Env.__new__(Env)
        env.action_space = 9
        env.combat_timeout = 120
        env._reset_combat_timer()
        env.reward_config = RewardConfig()
        env._episode_done = False
        env.render_enabled = False
        env.player = None
        env.new_level = False
        env.enemy_list = EnemyList()
        env.bullets = Bullets()
        env.falling_items = FallingItems()
        env.action_manager = Mock()
        env.action_manager.fire_reward = 0.0
        env.action_manager.get_fire_cool_down.return_value = 0.3
        env.fps_calculator = Mock()
        env.render = Mock()
        env.map = Map(render_enabled=False)
        env.map.reset(0)
        env.map.next_level()
        env.game_window = SimpleNamespace(
            update=Mock(return_value=0),
            player_box=DetectionResult(1, [0.4, 0.4, 0.45, 0.45]),
            enemy_boxes=[DetectionResult(1, [0.6, 0.6, 0.65, 0.65])],
            bullet_list_normal=[(0.3, 0.4, 0)],
            falling_items_normal=[], items={"pao": False, "dan": False, "fan": False},
            hit_point=4,
        )
        with patch.object(env.map, "update", side_effect=AssertionError("智能体不应计算危险图")), \
             patch.object(Enemy, "calculate_vision", side_effect=AssertionError("不应计算视野")), \
             patch.object(Enemy, "get_distance_array", side_effect=AssertionError("不应计算距离图")), \
             patch.object(env, "try_next_level", wraps=env.try_next_level) as finish:
            self.assertEqual(env.step(1), 0)
            state = env.state_encode()
            finish.assert_called_once_with()
        self.assertEqual(state.shape, (60,))
        self.assertEqual(state[17], 1)
        self.assertEqual(state[32], 1)
        self.assertIsNone(env.enemy_list[0].vision)
        self.assertIsNone(env.enemy_list[0].distance_array)

    def test_auto_path_builds_navigation_before_planning(self):
        env = Env.__new__(Env)
        env.player = Mock()
        env.bullets = Mock()
        env.enemy_list = Mock()
        env.map = Mock()
        env.map.get_shot_place_direction.return_value = (None, [], 0)
        env.move_index, env.move_end = None, None
        env.new_level = False
        env.can_next_level = Mock(return_value=False)
        env.action_manager = Mock()
        env.render = Mock()
        env.update = Mock(return_value=0)
        self.assertEqual(env.move_to_enemy_and_update(), 0)
        self.assertEqual(env.map.method_calls[0][0], "update")
        self.assertEqual(env.map.method_calls[1][0], "get_shot_place_direction")

    def test_enemy_navigation_is_lazy_and_cached(self):
        game_map = Map(render_enabled=False)
        game_map.reset(0)
        maze = game_map.get_maze()
        enemy = Enemy(DetectionResult(1, [0.6, 0.6, 0.65, 0.65]), maze)
        self.assertIsNone(enemy.distance_array)
        self.assertIsNone(enemy.vision)
        with patch.object(enemy, "get_distance_array", wraps=enemy.get_distance_array) as distance, \
             patch.object(enemy, "calculate_vision", wraps=enemy.calculate_vision) as vision:
            enemy.prepare_navigation(maze)
            enemy.prepare_navigation(maze)
            distance.assert_called_once_with()
            self.assertEqual(vision.call_count, 1)


if __name__ == "__main__":
    unittest.main()
