"""用模拟单调时钟验证引擎超时和逐关战斗预算，不等待真实时间。"""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from zmxy4_treasure_hunt.voyage.env import Env
from zmxy4_treasure_hunt.voyage.game_window import GameWindow
from zmxy4_treasure_hunt.voyage.core.onnx_model import DetectionResult


class TimeoutTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        clock = patch("zmxy4_treasure_hunt.voyage.env.time.monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)

    def make_env(self):
        env = Env.__new__(Env)
        env.combat_timeout = 120
        env.timeout = 5
        env.stay_timeout = 30
        env._reset_combat_timer()
        env.map = SimpleNamespace(
            level=1, get_maze=Mock(), fire_direction=Mock(return_value=None),
            get_next_direction=Mock(return_value=(None, [])),
        )
        env.enemy_list = MagicMock()
        env.enemy_list.__len__.return_value = 1
        env.bullets = MagicMock()
        env.bullets.__len__.return_value = 0
        env.bullets.new_bullets = []
        env.falling_items = Mock()
        env.game_window = SimpleNamespace(
            update=Mock(return_value=0), player_box=Mock(), items={"pao": False},
            enemy_boxes=[], bullet_list_normal=[], falling_items_normal=[],
        )
        env.action_manager = Mock()
        env.fps_calculator = Mock()
        env.player = None
        env.render_enabled = False
        env.new_level = False
        env.render = Mock()
        return env

    def make_env_with_missing_player(self):
        env = self.make_env()
        detections = {
            "player": [], "enemy": [], "bullet": [],
            "pao": [], "dan": [], "fan": [], "jia": [],
        }
        model = Mock(return_value=detections)
        with patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel", return_value=model):
            env.game_window = GameWindow(capture_enabled=False)
        self.enterContext(patch.object(env.game_window, "capture_screen",
                                      return_value=np.zeros((200, 320, 3), dtype=np.uint8)))
        self.enterContext(patch.object(env.game_window, "locate_map_contours", return_value=[object()]))
        self.enterContext(patch.object(env.game_window, "get_hit_point", return_value=4))
        return env, detections

    def test_first_player_detection_recovers_before_creating_player(self):
        env, detections = self.make_env_with_missing_player()
        box = DetectionResult(0.9, [0.2, 0.3, 0.4, 0.6])
        player = SimpleNamespace(is_next=lambda other: False)

        def wait(_):
            self.assertIsNone(env.player)
            self.now += 0.1
            detections["player"] = [box]

        with (
            patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait),
            patch("zmxy4_treasure_hunt.voyage.env.Player", return_value=player) as create_player,
        ):
            self.assertEqual(env.update(), 0)
        create_player.assert_called_once_with(box, env.map.get_maze.return_value)
        self.assertIs(env.player, player)

    def test_first_player_detection_timeout_never_creates_player(self):
        env, _ = self.make_env_with_missing_player()

        def wait(_):
            self.now += 1

        with (
            patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait),
            patch("zmxy4_treasure_hunt.voyage.env.Player") as create_player,
        ):
            self.assertEqual(env.update(), -4)
        create_player.assert_not_called()
        self.assertIsNone(env.player)
        env.map.get_maze.assert_not_called()
        self.assertEqual(env.combat_elapsed_seconds, 0)

    def test_deadline_same_level_and_reset_on_new_level(self):
        env = self.make_env()
        self.assertEqual(env._update_combat_timer(), 0)
        self.now = 119.99
        self.assertEqual(env._update_combat_timer(), 0)
        self.now = 120
        self.assertEqual(env._update_combat_timer(), -3)
        self.assertEqual(env.combat_elapsed_seconds, 120)
        self.now = 200
        self.assertEqual(env.combat_elapsed_seconds, 120)
        env.map.level = 2
        self.assertEqual(env._update_combat_timer(), 0)
        self.assertEqual(env.combat_elapsed_seconds, 0)
        env._reset_combat_timer()
        self.assertIsNone(env._combat_level)
        self.assertEqual(env.combat_elapsed_seconds, 0)

    def test_navigation_pause_does_not_reset_same_level_budget(self):
        env = self.make_env()
        env._update_combat_timer()
        self.now = 20
        env.enemy_list.__len__.return_value = 0
        env._update_combat_timer()
        self.now = 100
        self.assertEqual(env.combat_elapsed_seconds, 20)
        env.enemy_list.__len__.return_value = 1
        env._update_combat_timer()
        self.now = 200
        self.assertEqual(env._update_combat_timer(), -3)

    def test_start_area_is_not_combat_and_bullets_alone_count(self):
        env = self.make_env()
        env.map.level = 0
        env._update_combat_timer()
        self.now = 500
        self.assertEqual(env.combat_elapsed_seconds, 0)
        env.map.level = 1
        env.enemy_list.__len__.return_value = 0
        env.bullets.__len__.return_value = 1
        env._update_combat_timer()
        self.now = 620
        self.assertEqual(env._update_combat_timer(), -3)

    def test_loading_timeout_returns_engine_status(self):
        env = self.make_env()
        env._update_combat_timer()
        self.now = 10
        env.game_window.update.return_value = -1

        def wait(_):
            self.now += 1

        with patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait):
            self.assertEqual(env.update(), -4)
        self.assertEqual(env.combat_elapsed_seconds, 10)

    def test_loading_recovery_excludes_loading_time_and_combat_expires_in_update(self):
        env = self.make_env()
        env._update_combat_timer()
        self.now = 10
        env.game_window.update.side_effect = [-1, 0]

        def wait(_):
            self.now += 4

        player = SimpleNamespace(is_next=lambda other: False)
        with patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait), \
             patch("zmxy4_treasure_hunt.voyage.env.Player", return_value=player):
            self.assertEqual(env.update(), 0)
        self.assertEqual(env.combat_elapsed_seconds, 10)
        self.now = 124
        env.game_window.update.side_effect = None
        with patch("zmxy4_treasure_hunt.voyage.env.Player", return_value=player):
            self.assertEqual(env.update(), -3)

    def test_auto_start_propagates_engine_and_combat_status(self):
        for status in (-3, -4):
            env = self.make_env()
            env.new_level = True
            env.player = Mock()
            env.map.get_task = Mock(return_value=5)
            env.update = Mock(return_value=status)
            with patch("zmxy4_treasure_hunt.voyage.env.time.sleep"):
                self.assertEqual(env.try_start_level(), status)
            env.move_to_enemy_and_update = Mock(return_value=0)
            env.try_next_level = Mock()
            with patch("zmxy4_treasure_hunt.voyage.env.time.sleep"):
                self.assertEqual(env.auto_step(), status)
            env.try_next_level.assert_not_called()

    def test_reset_clears_timer_and_propagates_initial_engine_failure(self):
        env = self.make_env()
        env._update_combat_timer()
        self.now = 100
        env.update = Mock(return_value=-4)
        status = env.reset()
        self.assertEqual(status, -4)
        self.assertEqual(env.combat_elapsed_seconds, 0)


if __name__ == "__main__":
    unittest.main()
