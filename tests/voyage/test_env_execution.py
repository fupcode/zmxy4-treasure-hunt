"""验证环境执行、感知与导航的计算边界及动作状态。"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from zmxy4_treasure_hunt.voyage.core.action_manager import ActionManager
from zmxy4_treasure_hunt.voyage.core.direction import Direction
from zmxy4_treasure_hunt.voyage.core.entities import Enemy
from zmxy4_treasure_hunt.voyage.core.map import Map
from zmxy4_treasure_hunt.voyage.core.onnx_model import DetectionResult
from zmxy4_treasure_hunt.voyage.env import Voyager


class VoyagerExecutionTests(unittest.TestCase):
    def make_action_manager(self):
        self.enterContext(patch("zmxy4_treasure_hunt.voyage.core.action_manager.key_down"))
        self.enterContext(patch("zmxy4_treasure_hunt.voyage.core.action_manager.key_up"))
        self.enterContext(patch("zmxy4_treasure_hunt.voyage.core.action_manager.time.sleep"))
        manager = ActionManager()
        manager.fire = Mock()
        return manager

    def test_auto_run_returns_initial_failure_without_taking_actions(self):
        voyager = Voyager.__new__(Voyager)
        voyager.reset = Mock(return_value=-4)
        voyager.auto_step = Mock()
        self.assertEqual(voyager.auto_run(), -4)
        voyager.auto_step.assert_not_called()

    def test_auto_run_returns_terminal_status_after_automatic_steps(self):
        for status in (1, -2, -3, -4):
            with self.subTest(status=status):
                voyager = Voyager.__new__(Voyager)
                voyager.reset = Mock(return_value=0)
                voyager.auto_step = Mock(side_effect=[0, status])
                self.assertEqual(voyager.auto_run(), status)
                self.assertEqual(voyager.auto_step.call_count, 2)

    def test_terminal_update_skips_scene_processing(self):
        for status in (1, -2):
            with self.subTest(status=status):
                voyager = Voyager.__new__(Voyager)
                voyager._reset_combat_timer()
                voyager.fps_calculator = Mock()
                voyager.game_window = Mock()
                voyager.game_window.update.return_value = status
                voyager.map = Mock(level=4)
                voyager.map.get_maze.side_effect = AssertionError("终局不应更新地图")
                self.assertEqual(voyager.update(), status)
                voyager.map.get_maze.assert_not_called()

    def test_perception_update_never_builds_danger_map(self):
        for enemies, bullets in ((0, 0), (1, 0), (0, 1)):
            with self.subTest(enemies=enemies, bullets=bullets):
                voyager = Voyager.__new__(Voyager)
                voyager.combat_timeout = 120
                voyager._reset_combat_timer()
                voyager.player = None
                voyager.render_enabled = False
                voyager.new_level = False
                voyager.fps_calculator = Mock()
                voyager.game_window = Mock()
                voyager.game_window.update.return_value = 0
                voyager.game_window.items = {"pao": False}
                voyager.map = Mock(level=1)
                voyager.action_manager = Mock()
                voyager.enemy_list = MagicMock()
                voyager.enemy_list.__len__.return_value = enemies
                voyager.bullets = MagicMock()
                voyager.bullets.__len__.return_value = bullets
                voyager.bullets.new_bullets = []
                voyager.falling_items = Mock()
                with patch("zmxy4_treasure_hunt.voyage.env.Player"):
                    self.assertEqual(voyager.update(), 0)
                if enemies or bullets:
                    voyager.fps_calculator.trigger.assert_called_once_with(log_to_terminal=True)
                else:
                    voyager.fps_calculator.trigger.assert_not_called()
                    voyager.fps_calculator.reset.assert_called_once_with()
                voyager.map.update.assert_not_called()
                self.assertIsNone(voyager.map.debug_image)
                self.assertIsNone(voyager.map.flee_mazes)

    def test_auto_path_builds_navigation_before_planning(self):
        voyager = Voyager.__new__(Voyager)
        voyager.player = Mock()
        voyager.bullets = Mock()
        voyager.enemy_list = Mock()
        voyager.map = Mock()
        voyager.map.get_shot_place_direction.return_value = (None, [], 0)
        voyager.new_level = False
        voyager.can_next_level = Mock(return_value=False)
        voyager.action_manager = self.make_action_manager()
        voyager.action_manager.last_direction = Direction.Right
        voyager.render = Mock()
        voyager.update = Mock(return_value=0)
        self.assertEqual(voyager.move_to_enemy_and_update(), 0)
        self.assertEqual(voyager.map.method_calls[0][0], "update")
        self.assertEqual(voyager.map.method_calls[1][0], "get_shot_place_direction")
        voyager.map.get_shot_place_direction.assert_called_once_with(
            voyager.player, last_direction=Direction.Right)
        self.assertIsNone(voyager.action_manager.last_direction)  # 本次执行了停留。

    def test_executed_actions_update_direction_and_stop_clears_it(self):
        voyager = Voyager.__new__(Voyager)
        voyager.action_manager = self.make_action_manager()
        voyager.action_manager.take_action_direction(Direction.DownLeft)
        self.assertEqual(voyager.action_manager.last_direction, Direction.DownLeft)
        voyager.action_manager.take_action_direction(None)
        self.assertIsNone(voyager.action_manager.last_direction)

    def test_stop_moving_releases_keys_without_firing(self):
        manager = self.make_action_manager()
        manager.last_direction = Direction.DownLeft
        with patch("zmxy4_treasure_hunt.voyage.core.action_manager.key_up") as key_up:
            manager.stop_moving()
        key_up.assert_called_once_with(Direction.DownLeft.value)
        self.assertIsNone(manager.last_direction)
        manager.fire.assert_not_called()

    def test_debug_preview_preserves_last_executed_direction(self):
        voyager = Voyager.__new__(Voyager)
        voyager.player = Mock()
        voyager.action_manager = Mock(last_direction=Direction.Right)
        voyager.update = Mock(return_value=0)
        voyager.update_navigation = Mock()
        voyager.map = Mock()
        voyager.map.get_shot_place_direction.return_value = (Direction.Left, [(1, 1)], 1)
        voyager.render = Mock()
        self.assertEqual(voyager.debug_step(), 0)
        self.assertEqual(voyager.action_manager.last_direction, Direction.Right)
        voyager.map.get_shot_place_direction.assert_called_once_with(
            voyager.player, last_direction=Direction.Right)
        voyager.action_manager.take_action_direction.assert_not_called()

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
