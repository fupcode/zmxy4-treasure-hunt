"""用模拟单调时钟验证引擎超时和逐关战斗预算，不等待真实时间。"""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from zmxy4_treasure_hunt.voyage.env import Voyager
from zmxy4_treasure_hunt.voyage.game_window import GameWindow
from zmxy4_treasure_hunt.voyage.core.onnx_model import DetectionResult


class TimeoutTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        clock = patch("zmxy4_treasure_hunt.voyage.env.time.monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)

    def make_voyager(self):
        voyager = Voyager.__new__(Voyager)
        voyager.combat_timeout = 120
        voyager.timeout = 5
        voyager.stay_timeout = 30
        voyager._reset_combat_timer()
        voyager.map = SimpleNamespace(
            level=1, get_maze=Mock(), fire_direction=Mock(return_value=None),
            get_next_direction=Mock(return_value=(None, [])),
        )
        voyager.enemy_list = MagicMock()
        voyager.enemy_list.__len__.return_value = 1
        voyager.bullets = MagicMock()
        voyager.bullets.__len__.return_value = 0
        voyager.bullets.new_bullets = []
        voyager.falling_items = Mock()
        voyager.game_window = SimpleNamespace(
            update=Mock(return_value=0), player_box=Mock(), items={"pao": False},
            enemy_boxes=[], bullet_list_normal=[], falling_items_normal=[],
        )
        voyager.action_manager = Mock()
        voyager.fps_calculator = Mock()
        voyager.player = None
        voyager.render_enabled = False
        voyager.new_level = False
        voyager.render = Mock()
        return voyager

    def make_voyager_with_missing_player(self):
        voyager = self.make_voyager()
        detections = {
            "player": [], "enemy": [], "bullet": [],
            "pao": [], "dan": [], "fan": [], "jia": [],
        }
        model = Mock(return_value=detections)
        with patch("zmxy4_treasure_hunt.voyage.game_window.OnnxModel", return_value=model):
            voyager.game_window = GameWindow(capture_enabled=False)
        self.enterContext(patch.object(voyager.game_window, "capture_screen",
                                      return_value=np.zeros((200, 320, 3), dtype=np.uint8)))
        self.enterContext(patch.object(voyager.game_window, "locate_map_contours", return_value=[object()]))
        self.enterContext(patch.object(voyager.game_window, "get_hit_point", return_value=4))
        return voyager, detections

    def test_first_player_detection_recovers_before_creating_player(self):
        voyager, detections = self.make_voyager_with_missing_player()
        box = DetectionResult(0.9, [0.2, 0.3, 0.4, 0.6])
        player = SimpleNamespace(is_next=lambda other: False)

        def wait(_):
            self.assertIsNone(voyager.player)
            self.now += 0.1
            detections["player"] = [box]

        with (
            patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait),
            patch("zmxy4_treasure_hunt.voyage.env.Player", return_value=player) as create_player,
        ):
            self.assertEqual(voyager.update(), 0)
        create_player.assert_called_once_with(box, voyager.map.get_maze.return_value)
        self.assertIs(voyager.player, player)

    def test_first_player_detection_timeout_never_creates_player(self):
        voyager, _ = self.make_voyager_with_missing_player()

        def wait(_):
            self.now += 1

        with (
            patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait),
            patch("zmxy4_treasure_hunt.voyage.env.Player") as create_player,
        ):
            self.assertEqual(voyager.update(), -4)
        create_player.assert_not_called()
        self.assertIsNone(voyager.player)
        voyager.map.get_maze.assert_not_called()
        self.assertEqual(voyager.combat_elapsed_seconds, 0)

    def test_deadline_same_level_and_reset_on_new_level(self):
        voyager = self.make_voyager()
        self.assertEqual(voyager._update_combat_timer(), 0)
        self.now = 119.99
        self.assertEqual(voyager._update_combat_timer(), 0)
        self.now = 120
        self.assertEqual(voyager._update_combat_timer(), -3)
        self.assertEqual(voyager.combat_elapsed_seconds, 120)
        self.now = 200
        self.assertEqual(voyager.combat_elapsed_seconds, 120)
        voyager.map.level = 2
        self.assertEqual(voyager._update_combat_timer(), 0)
        self.assertEqual(voyager.combat_elapsed_seconds, 0)
        voyager._reset_combat_timer()
        self.assertIsNone(voyager._combat_level)
        self.assertEqual(voyager.combat_elapsed_seconds, 0)

    def test_navigation_pause_does_not_reset_same_level_budget(self):
        voyager = self.make_voyager()
        voyager._update_combat_timer()
        self.now = 20
        voyager.enemy_list.__len__.return_value = 0
        voyager._update_combat_timer()
        self.now = 100
        self.assertEqual(voyager.combat_elapsed_seconds, 20)
        voyager.enemy_list.__len__.return_value = 1
        voyager._update_combat_timer()
        self.now = 200
        self.assertEqual(voyager._update_combat_timer(), -3)

    def test_start_area_is_not_combat_and_bullets_alone_count(self):
        voyager = self.make_voyager()
        voyager.map.level = 0
        voyager._update_combat_timer()
        self.now = 500
        self.assertEqual(voyager.combat_elapsed_seconds, 0)
        voyager.map.level = 1
        voyager.enemy_list.__len__.return_value = 0
        voyager.bullets.__len__.return_value = 1
        voyager._update_combat_timer()
        self.now = 620
        self.assertEqual(voyager._update_combat_timer(), -3)

    def test_loading_timeout_returns_engine_status(self):
        voyager = self.make_voyager()
        voyager._update_combat_timer()
        self.now = 10
        voyager.game_window.update.return_value = -1

        def wait(_):
            self.now += 1

        with patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait):
            self.assertEqual(voyager.update(), -4)
        self.assertEqual(voyager.combat_elapsed_seconds, 10)

    def test_loading_recovery_excludes_loading_time_and_combat_expires_in_update(self):
        voyager = self.make_voyager()
        voyager._update_combat_timer()
        self.now = 10
        voyager.game_window.update.side_effect = [-1, 0]

        def wait(_):
            self.now += 4

        player = SimpleNamespace(is_next=lambda other: False)
        with patch("zmxy4_treasure_hunt.voyage.env.time.sleep", side_effect=wait), \
             patch("zmxy4_treasure_hunt.voyage.env.Player", return_value=player):
            self.assertEqual(voyager.update(), 0)
        self.assertEqual(voyager.combat_elapsed_seconds, 10)
        self.now = 124
        voyager.game_window.update.side_effect = None
        with patch("zmxy4_treasure_hunt.voyage.env.Player", return_value=player):
            self.assertEqual(voyager.update(), -3)

    def test_auto_start_propagates_engine_and_combat_status(self):
        for status in (-3, -4):
            voyager = self.make_voyager()
            voyager.new_level = True
            voyager.player = Mock()
            voyager.map.get_task = Mock(return_value=5)
            voyager.update = Mock(return_value=status)
            with patch("zmxy4_treasure_hunt.voyage.env.time.sleep"):
                self.assertEqual(voyager.try_start_level(), status)
            voyager.move_to_enemy_and_update = Mock(return_value=0)
            voyager.try_next_level = Mock()
            with patch("zmxy4_treasure_hunt.voyage.env.time.sleep"):
                self.assertEqual(voyager.auto_step(), status)
            voyager.try_next_level.assert_not_called()

    def test_reset_clears_timer_and_propagates_initial_engine_failure(self):
        voyager = self.make_voyager()
        voyager._update_combat_timer()
        self.now = 100
        voyager.update = Mock(return_value=-4)
        status = voyager.reset()
        self.assertEqual(status, -4)
        self.assertEqual(voyager.combat_elapsed_seconds, 0)


if __name__ == "__main__":
    unittest.main()
