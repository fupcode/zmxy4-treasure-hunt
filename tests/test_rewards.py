"""合成环境验证奖励归因与碰撞；不启动游戏，不发送按键。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from zaomeng_hanghai.core.action_manager import ActionManager
from zaomeng_hanghai.core.constants import MAZE_HEIGHT, MAZE_WIDTH
from zaomeng_hanghai.core.direction import Direction, move_list
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.rewards import RewardConfig, calculate_reward
from zaomeng_hanghai.env import Env
from zaomeng_hanghai.core.observation import OBSERVATION_DIM


def empty_map():
    game_map = Map.__new__(Map)
    game_map.level = 1
    game_map.level_mazes = [np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool) for _ in range(4)]
    game_map.map_template_without_exit = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
    return game_map


class CollisionTests(unittest.TestCase):
    def setUp(self):
        self.game_map = empty_map()
        self.player = SimpleNamespace(x=100, y=100)

    def test_stop_and_free_movement(self):
        for action in range(9):
            self.assertFalse(self.game_map.is_move_collision(self.player, action))
        self.game_map.get_maze()[99, 99] = True
        self.assertFalse(self.game_map.is_move_collision(self.player, 0))

    def test_thin_wall_and_diagonal_corner(self):
        self.game_map.get_maze()[100, 102] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 1))
        self.assertFalse(self.game_map.is_move_collision(self.player, 3))
        self.game_map.get_maze()[100, 101] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 1))
        self.assertTrue(self.game_map.is_move_collision(self.player, 5))
        self.game_map.get_maze()[100, 101] = False
        self.game_map.get_maze()[101, 100] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 5))

    def test_bounds_and_common_forbidden_region(self):
        for x, y, action in ((0, 100, 3), (319, 100, 1), (100, 0, 4), (100, 199, 2)):
            self.assertTrue(self.game_map.is_move_collision(SimpleNamespace(x=x, y=y), action))
        self.game_map.map_template_without_exit[100, 101] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 1))
        for action in (-1, 9, 0.5):
            with self.assertRaises(ValueError):
                self.game_map.is_move_collision(self.player, action)

    def test_all_directions_check_three_pixels(self):
        maze = self.game_map.get_maze()
        for direction, (dx, dy) in enumerate(move_list, 1):
            for distance in (1, 2, 3):
                maze.fill(False)
                maze[100 + dy * distance, 100 + dx * distance] = True
                self.assertTrue(self.game_map.is_move_collision(self.player, direction))
            maze.fill(False)
            maze[100 + dy * 4, 100 + dx * 4] = True
            self.assertFalse(self.game_map.is_move_collision(self.player, direction))
        maze.fill(False)
        maze[102, 101] = True
        self.assertTrue(self.game_map.is_move_collision(self.player, 5))


class RewardTests(unittest.TestCase):
    def make_env(self, fire_reward=0.0):
        env = Env.__new__(Env)
        env.action_space = 9
        env.combat_timeout = 120
        env._reset_combat_timer()
        env.reward_config = RewardConfig()
        env._episode_done = False
        env.player = SimpleNamespace(x=100, y=100, xn=100 / MAZE_WIDTH, yn=100 / MAZE_HEIGHT)
        env.map = empty_map()
        env.game_window = SimpleNamespace(hit_point=4)
        env.action_manager = SimpleNamespace(take_action=Mock(), fire_reward=fire_reward)
        env.render = Mock()
        env.update = Mock(return_value=0)
        env.try_next_level = Mock(return_value=0)
        env.state_encode = Mock(return_value=np.zeros(OBSERVATION_DIM, dtype=np.float32))
        env.new_level = False
        return env

    def test_existing_fire_rewards_unchanged(self):
        for ready, direction, expected in (
            (True, Direction.Right, 1), (True, None, -0.05),
            (False, Direction.Right, -0.1), (False, None, 0.02),
        ):
            manager = ActionManager()
            manager.fire_direction = direction
            with patch.object(manager, "_can_fire", return_value=ready), \
                 patch("zaomeng_hanghai.core.action_manager.move_to"), \
                 patch("zaomeng_hanghai.core.action_manager.stop_to"), \
                 patch("zaomeng_hanghai.core.action_manager.key_down"), \
                 patch("zaomeng_hanghai.core.action_manager.key_up"), \
                 patch("zaomeng_hanghai.core.action_manager.time.sleep"):
                fired = manager.fire()
            self.assertEqual(manager.fire_reward, expected)
            self.assertEqual(fired, int(ready and direction is not None))
            manager.reset()
            self.assertEqual(manager.fire_reward, 0)

    def test_stationary_is_small_and_cooldown_cannot_pay_for_waiting(self):
        env = self.make_env(fire_reward=0.02)
        self.assertEqual(env.step(0), 0)
        self.assertAlmostEqual(env.reward_terms["stationary"], -0.005)
        self.assertLess(env.last_reward, 0)
        self.assertEqual(env.reward_terms["wall_collision"], 0)
        # 选移动但实际没走，也要算静止。
        env.step(1)
        self.assertEqual(env.reward_terms["stationary"], -0.005)

    def test_free_movement_and_wall_collision(self):
        env = self.make_env(fire_reward=1)

        def move():
            env.player.x += 3
            env.player.xn += 3 / MAZE_WIDTH
            return 0

        env.update.side_effect = move
        env.step(1)
        self.assertAlmostEqual(env.last_reward, 0.98)
        self.assertEqual(env.reward_terms["stationary"], 0)
        env.map.get_maze()[100, env.player.x + 1] = True
        env.update.side_effect = None
        env.step(1)
        self.assertEqual(env.reward_terms["wall_collision"], -0.2)
        self.assertEqual(env.reward_terms["stationary"], 0)
        self.assertAlmostEqual(env.last_reward, 0.78)

    def test_navigation_cannot_overwrite_agent_fire_or_movement(self):
        env = self.make_env(fire_reward=1)

        def navigate():
            env.action_manager.fire_reward = -0.05
            env.player.xn = 0.8
            env.map.level = 2
            return 0

        env.try_next_level.side_effect = navigate
        env.step(0)
        self.assertEqual(env.reward_terms["fire"], 1)
        self.assertEqual(env.reward_terms["stationary"], -0.005)
        self.assertEqual(env.reward_terms["next_level"], 5)
        self.assertAlmostEqual(env.last_reward, 5.975)

    def test_damage_and_terminal_screenshot(self):
        env = self.make_env()
        env.update.side_effect = lambda: (setattr(env.game_window, "hit_point", 3) or 0)
        env.step(0)
        self.assertEqual(env.reward_terms["damage"], -5)
        # 终局没有可信的 HP 图像，不能因读到 0 再额外扣剩余血量。
        env.update.side_effect = lambda: (setattr(env.game_window, "hit_point", 0) or -2)
        _, reward, terminated, truncated, info = env.step_with_reward(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reward_terms"]["damage"], 0)
        self.assertEqual(info["reward_terms"]["death"], -20)
        self.assertAlmostEqual(reward, -20.02)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_success_timeout_and_reward_wrapper(self):
        for status, term, trunc, name, amount in (
            (1, True, False, "success", 30), (-3, False, True, "timeout", -10)
        ):
            env = self.make_env()
            env.update.return_value = status
            obs, reward, terminated, truncated, info = env.step_with_reward(0)
            self.assertEqual(obs.shape, (OBSERVATION_DIM,))
            self.assertEqual((terminated, truncated), (term, trunc))
            self.assertTrue(info["transition_valid"])
            self.assertEqual(info["reward_terms"][name], amount)
            self.assertAlmostEqual(reward, amount - 0.02)
            env.action_manager.take_action.assert_called_once_with(0)
            env.try_next_level.assert_not_called()
        env = self.make_env()
        env.update.side_effect = TimeoutError("Loading timeout")
        self.assertEqual(env.step(0), -4)
        self.assertEqual(env.last_step_info["error"], "Loading timeout")
        self.assertEqual(env.reward_terms["timeout"], 0)
        self.assertEqual(env.last_reward, 0)
        self.assertFalse(env.last_step_info["transition_valid"])

    def test_engine_timeout_discards_every_reward_component(self):
        for source in ("update", "try_next_level"):
            env = self.make_env(fire_reward=1)
            env.map.get_maze()[100, 101] = True
            getattr(env, source).return_value = -4
            _, reward, terminated, truncated, info = env.step_with_reward(1)
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertFalse(info["transition_valid"])
            self.assertTrue(all(value == 0 for value in info["reward_terms"].values()))

    def test_level_transition_does_not_count_as_stationary_or_damage_healing(self):
        env = self.make_env()

        def enter_level():
            env.map.level = 2
            return 0

        env.update.side_effect = enter_level
        env.step(0)
        self.assertEqual(env.reward_terms["stationary"], 0)
        self.assertEqual(env.reward_terms["next_level"], 5)
        env.game_window.hit_point = 3
        env.update.side_effect = lambda: (setattr(env.game_window, "hit_point", 4) or 0)
        env.step(0)
        self.assertEqual(env.reward_terms["damage"], 0)

    def test_reward_configuration_and_invalid_actions(self):
        env = self.make_env()
        env.reward_config = RewardConfig(stationary=-0.001)
        env.step(np.int64(0))
        self.assertEqual(env.reward_terms["stationary"], -0.001)
        for action in (-1, 9, 1.5):
            with self.assertRaises(ValueError):
                env.step(action)
        terms = calculate_reward(
            RewardConfig(), fire_reward=1, stationary=False, wall_collision=False,
            hp_lost=2, levels_passed=1, status=0,
        )
        self.assertEqual(terms["damage"], -10)
        self.assertEqual(terms["next_level"], 5)


if __name__ == "__main__":
    unittest.main()
