"""验证感知更新、自动战斗寻路和过关接管的计算边界。"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from zaomeng_hanghai.core.direction import Direction
from zaomeng_hanghai.core.entities import Enemy
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.env import Env


class NextLevelTests(unittest.TestCase):
    def test_terminal_update_skips_scene_processing(self):
        for status in (1, -2):
            with self.subTest(status=status):
                env = Env.__new__(Env)
                env._reset_combat_timer()
                env.game_window = Mock()
                env.game_window.update.return_value = status
                env.map = Mock(level=4)
                env.map.get_maze.side_effect = AssertionError("终局不应更新地图")
                self.assertEqual(env.update(), status)
                env.map.get_maze.assert_not_called()

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

    def test_auto_path_builds_navigation_before_planning(self):
        env = Env.__new__(Env)
        env.player = Mock()
        env.bullets = Mock()
        env.enemy_list = Mock()
        env.map = Mock()
        env.map.get_shot_place_direction.return_value = (None, [], 0)
        env.last_direction = Direction.Right
        env.new_level = False
        env.can_next_level = Mock(return_value=False)
        env.action_manager = Mock()
        env.render = Mock()
        env.update = Mock(return_value=0)
        self.assertEqual(env.move_to_enemy_and_update(), 0)
        self.assertEqual(env.map.method_calls[0][0], "update")
        self.assertEqual(env.map.method_calls[1][0], "get_shot_place_direction")
        env.map.get_shot_place_direction.assert_called_once_with(
            env.player, last_direction=Direction.Right)
        self.assertIsNone(env.last_direction)  # 本次执行了停留。

    def test_executed_actions_update_direction_and_stop_clears_it(self):
        env = Env.__new__(Env)
        env.last_direction = None
        env.action_manager = Mock()
        env._take_expert_action(Direction.DownLeft, 'combat')
        self.assertEqual(env.last_direction, Direction.DownLeft)
        env.action_manager.take_action_direction.assert_called_once_with(Direction.DownLeft)
        env._take_expert_action(None, 'combat')
        self.assertIsNone(env.last_direction)

    def test_debug_preview_preserves_last_executed_direction(self):
        env = Env.__new__(Env)
        env.last_direction = Direction.Right
        env.player = Mock()
        env.action_manager = Mock()
        env.update = Mock(return_value=0)
        env.update_navigation = Mock()
        env.map = Mock()
        env.map.get_shot_place_direction.return_value = (Direction.Left, [(1, 1)], 1)
        env.render = Mock()
        self.assertEqual(env.debug_step(), 0)
        self.assertEqual(env.last_direction, Direction.Right)
        env.map.get_shot_place_direction.assert_called_once_with(
            env.player, last_direction=Direction.Right)
        env.action_manager.take_action_direction.assert_not_called()

    def test_enemy_navigation_is_lazy_and_cached(self):
        game_map = Map(render_enabled=False)
        game_map.reset(0)
        maze = game_map.get_maze()
        enemy = Enemy(DetectionResult(1, [0.6, 0.6, 0.65, 0.65]), maze)
        self.assertIsNone(enemy.distance_array)
        self.assertIsNone(enemy.fire_range)
        with patch.object(enemy, "get_distance_array", wraps=enemy.get_distance_array) as distance, \
             patch.object(enemy, "calculate_fire_range", wraps=enemy.calculate_fire_range) as fire_range:
            enemy.prepare_navigation(maze)
            enemy.prepare_navigation(maze)
            distance.assert_called_once_with()
            self.assertEqual(fire_range.call_count, 1)


if __name__ == "__main__":
    unittest.main()
