"""核心模块拆分后的行为验证；不操作真实窗口或键盘。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from zaomeng_hanghai.core import action_manager
from zaomeng_hanghai.core.action_manager import ActionManager
from zaomeng_hanghai.core.constants import MAZE_HEIGHT, MAZE_WIDTH
from zaomeng_hanghai.core.direction import Direction
from zaomeng_hanghai.core.entities import EnemyList, Player
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.onnx_model import DetectionResult
from zaomeng_hanghai.env import Env


class CoreGameplayTests(unittest.TestCase):
    def test_enemy_tracking_retains_fire_state(self):
        maze = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
        box = DetectionResult(0.9, [0.2, 0.2, 0.3, 0.3])
        enemies = EnemyList()
        enemies.update([box], maze)
        enemies[0].set_fired(100, 2)
        enemies.update([box], maze)
        self.assertEqual(len(enemies), 1)
        self.assertTrue(enemies[0].is_safe(101))
        self.assertFalse(enemies[0].is_safe(103))

    def test_fire_temporarily_turns_and_restores_movement(self):
        manager = ActionManager()
        manager.last_direction = Direction.Right
        manager.last_fire_time = 0
        events = Mock()
        with patch.multiple(
            action_manager, stop_to=events.stop_to, move_to=events.move_to,
            key_down=events.key_down, key_up=events.key_up,
        ), patch.object(action_manager.time, "time", return_value=10), patch.object(
            action_manager.time, "sleep", events.sleep,
        ):
            self.assertEqual(manager.fire_to(Direction.Up), 1)
            self.assertEqual(manager.fire_to(Direction.Up), 0)
        self.assertEqual(events.mock_calls, [
            call.stop_to(Direction.Right), call.move_to(Direction.Up),
            call.sleep(0.002), call.key_down("J"), call.key_up("J"),
            call.stop_to(Direction.Up), call.move_to(Direction.Right),
        ])
        self.assertEqual(manager.last_direction, Direction.Right)

    def test_map_reset_and_shared_direction(self):
        game_map = Map(180)
        game_map.reset(0)
        self.assertEqual(game_map.get_maze().shape, (MAZE_HEIGHT, MAZE_WIDTH))
        game_map.next_level()
        self.assertEqual(game_map.get_task(), game_map.tasks_list[0][0])
        self.assertEqual(game_map.get_maze().shape, (MAZE_HEIGHT, MAZE_WIDTH))
        enemy = object()
        with patch.object(game_map, "fire_distance", return_value=(10, 1)):
            self.assertIs(game_map.fire_direction(object(), [enemy]), Direction.Right)

    def test_env_connects_moved_entities_and_map(self):
        # 绕过窗口初始化，验证实际 Env.update 调用拆分后的实体与地图接口。
        env = Env.__new__(Env)
        env.wait_loading = Mock(return_value=0)
        env.map = Mock(level=1)
        env.map.get_maze.return_value = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
        env.player = None
        env.enemy_list = EnemyList()
        env.bullets = Mock(new_bullets=[])
        env.action_manager = Mock()
        env.falling_items = Mock()
        env.game_window = SimpleNamespace(
            player_box=DetectionResult(0.9, [0.5, 0.5, 0.6, 0.6]),
            enemy_boxes=[DetectionResult(0.9, [0.2, 0.2, 0.3, 0.3])],
            bullet_list_normal=[], items={"pao": False}, falling_items_normal=[],
        )
        self.assertEqual(env.update(), 0)
        self.assertIsInstance(env.player, Player)
        self.assertEqual(len(env.enemy_list), 1)
        env.map.update.assert_called_once_with(env.player, env.bullets, env.enemy_list)


if __name__ == "__main__":
    unittest.main()
