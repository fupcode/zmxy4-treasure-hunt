"""合成轨迹测试 PPO、故障边界、奖励去重和完整续训，不连接游戏。"""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from zaomeng_hanghai.policy import ACTION_NAMES, ActorCritic, load_policy
from zaomeng_hanghai.core.observation import OBSERVATION_DIM, OBSERVATION_SCHEMA_VERSION
from zaomeng_hanghai.train.ppo import (
    PPOConfig, PPOTrainer, TrainingState, Transition, compute_gae, restore_checkpoint, save_checkpoint,
)
from zaomeng_hanghai.train.train_ppo import RewardChangeLogger, build_parser, prepare_training, run_training


class FakeEnv:
    def __init__(self, episodes):
        self.episodes = iter(episodes)
        self.game_window = SimpleNamespace(recover=Mock())
        self.index = 0

    def observation(self):
        obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        obs[0] = self.index / 10
        return obs

    def reset(self):
        self.current = iter(next(self.episodes))
        self.index += 1
        return self.observation(), 0

    def step_with_reward(self, action):
        item = next(self.current)
        if isinstance(item, BaseException):
            raise item
        reward, status = item
        return self.observation(), reward, status in (1, -2), status in (-3, -4), {
            "status_code": status, "transition_valid": status != -4, "reward_terms": {"fire": reward},
        }


class PPOTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "runs"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_ppo_", dir=scratch)
        self.root = Path(self.temporary.name).resolve()
        if not self.root.is_relative_to(scratch.resolve()):
            raise ValueError("Test temporary directory must stay inside workspace runs/")
        self.addCleanup(self.temporary.cleanup)
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.env_config = {"timeout": 5, "stay_timeout": 30, "combat_timeout": 120}

    def trainer(self, **kwargs):
        return PPOTrainer(ActorCritic(hidden_sizes=(16,)), PPOConfig(**kwargs))

    def sample_bandit(self, trainer, count=256):
        obs = torch.zeros(count, OBSERVATION_DIM)
        with torch.no_grad():
            actions, log_probs, values = trainer.model.act(obs)
        return [Transition(
            obs[i].numpy(), int(actions[i]), float(log_probs[i]), float(values[i]),
            1.0 if actions[i] == 2 else -1.0, 0.0, True,
        ) for i in range(count)]

    def test_gae_bootstrap_and_trace_boundaries(self):
        # 第一个片段结束时 bootstrap V=10，但不能递推到后一局的奖励 100。
        rewards = torch.tensor([1.0, 2.0, 100.0])
        values = torch.tensor([0.5, 1.0, 0.0])
        next_values = torch.tensor([1.0, 10.0, 0.0])
        ends = torch.tensor([False, True, True])
        advantages, returns = compute_gae(rewards, values, next_values, ends, 0.9, 0.8)
        torch.testing.assert_close(advantages, torch.tensor([8.6, 10.0, 100.0]))
        torch.testing.assert_close(returns, torch.tensor([9.1, 11.0, 100.0]))

    def test_ppo_updates_actor_and_critic_and_improves_rewarded_action(self):
        trainer = self.trainer(learning_rate=0.01, update_epochs=5, batch_size=64, target_kl=0)
        samples = self.sample_bandit(trainer)
        before_critic = [parameter.clone() for parameter in trainer.model.critic.parameters()]
        with torch.no_grad():
            before = trainer.model.actor(torch.zeros(OBSERVATION_DIM)).softmax(-1)[2].item()
        metrics = trainer.update(samples)
        with torch.no_grad():
            after = trainer.model.actor(torch.zeros(OBSERVATION_DIM)).softmax(-1)[2].item()
        self.assertGreater(after, before)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before_critic, trainer.model.critic.parameters())))
        self.assertEqual(metrics["samples"], 256)
        self.assertTrue(all(np.isfinite(metrics[key]) for key in ("policy_loss", "value_loss", "entropy", "approx_kl")))

    def test_resume_restores_optimizer_rng_progress_and_inference(self):
        trainer = self.trainer(update_epochs=2, batch_size=64, target_kl=0)
        samples = self.sample_bandit(trainer)
        trainer.update(samples)
        state = TrainingState(env_steps=256, trained_steps=256, updates=1, episodes=3, last_logged_reward=1)
        path = self.root / "last.pt"
        save_checkpoint(path, trainer, state, self.env_config)
        expected_random = (random.random(), np.random.random(), torch.rand(3))
        restored, restored_state, environment = restore_checkpoint(path)
        self.assertEqual(restored_state, state)
        self.assertEqual(environment, self.env_config)
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.random(), expected_random[1])
        torch.testing.assert_close(torch.rand(3), expected_random[2])
        for left, right in zip(trainer.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue(restored.optimizer.state)
        torch.manual_seed(99)
        trainer.update(samples)
        torch.manual_seed(99)
        restored.update(samples)
        for left, right in zip(trainer.model.parameters(), restored.model.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        policy = load_policy(path)
        self.assertIn(policy.predict(np.zeros(OBSERVATION_DIM, dtype=np.float32)), range(9))
        override, _, _ = restore_checkpoint(path, overrides={"learning_rate": 0.002})
        self.assertEqual(override.optimizer.param_groups[0]["lr"], 0.002)

    def test_reward_logs_only_on_changes_and_keep_previous_across_sessions(self):
        state = TrainingState()
        path = self.root / "rewards.jsonl"
        logger = RewardChangeLogger(state, path)
        output = StringIO()
        with redirect_stdout(output):
            for reward in (1, 1, 0, 0, -0.005, -0.005, 1):
                logger.log(reward, 0, {"reward_terms": {"test": reward}})
            self.assertFalse(RewardChangeLogger(state, path).log(1, 0, {}))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["reward"] for record in records], [1, 0, -0.005, 1])
        self.assertEqual(output.getvalue().count("[Reward]"), 4)
        # 存档早于最近日志时，避免续训首项重复打印。
        with redirect_stdout(StringIO()):
            self.assertFalse(RewardChangeLogger(TrainingState(last_logged_reward=0), path).log(1, 0, {}))

    def test_reward_log_ignores_combat_time_changes_across_sessions(self):
        path = self.root / "rewards.jsonl"
        logger = RewardChangeLogger(TrainingState(), path)
        with redirect_stdout(StringIO()):
            self.assertTrue(logger.log(-0.05783333, 6, {
                "reward_terms": {"fire": -0.05, "combat_time": -0.00783333},
            }))
            self.assertFalse(logger.log(-0.0604167, 8, {
                "reward_terms": {"fire": -0.05, "combat_time": -0.0104167},
            }))
            resumed = RewardChangeLogger(TrainingState(), path)
            self.assertFalse(resumed.log(-0.062, 0, {
                "reward_terms": {"fire": -0.05, "combat_time": -0.012},
            }))
            self.assertTrue(resumed.log(-0.012, 0, {
                "reward_terms": {"fire": 0.0, "combat_time": -0.012},
            }))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["reward"], -0.05783333)
        self.assertEqual(records[0]["reward_terms"]["combat_time"], -0.00783333)

    def test_collection_discards_engine_step_and_resumes_for_additional_steps(self):
        trainer = self.trainer(rollout_steps=8, update_epochs=1)
        state = TrainingState()
        env = FakeEnv([[(1, 0), (1, 0), (0, -4)], [(-10, -3)]])
        captured = []
        original_update = trainer.update

        def update(samples):
            captured.extend(samples)
            return original_update(samples)

        trainer.update = update
        released = Mock()
        with redirect_stdout(StringIO()):
            run_training(env, trainer, state, self.root, steps=3, env_config=self.env_config, release_controls=released)
        self.assertEqual((state.env_steps, state.trained_steps, state.episodes, state.updates), (3, 3, 2, 1))
        self.assertEqual(state.engine_failures, 1)
        self.assertEqual([sample.reward for sample in captured], [1, 1, -10])
        self.assertEqual([sample.trace_end for sample in captured], [False, True, True])
        # -3 是截断，保留有效末帧的价值；引擎故障不产生一个伪造的零奖励转移。
        with torch.no_grad():
            # 用更新前的 next_value 仅断言它不是为截断硬置零。
            self.assertNotEqual(captured[-1].next_value, 0)
        self.assertGreaterEqual(released.call_count, 3)
        restored, state, environment = restore_checkpoint(self.root / "last.pt")
        with redirect_stdout(StringIO()):
            run_training(FakeEnv([[(1, 1)], [(1, 1)]]), restored, state, self.root,
                         steps=2, env_config=environment)
        self.assertEqual((state.env_steps, state.trained_steps, state.updates), (5, 5, 2))
        self.assertEqual(state.successes, 2)
        self.assertTrue((self.root / "best.pt").is_file())

    def test_keyboard_interrupt_keeps_last_complete_update(self):
        trainer = self.trainer(rollout_steps=8)
        release = Mock()
        with redirect_stdout(StringIO()):
            run_training(FakeEnv([[(1, 0), KeyboardInterrupt()]]), trainer, TrainingState(), self.root,
                         steps=8, env_config=self.env_config, release_controls=release)
        _, saved, _ = restore_checkpoint(self.root / "last.pt")
        self.assertEqual(saved.env_steps, 0)
        self.assertEqual(saved.updates, 0)
        release.assert_called()

    def test_repeated_engine_failures_stop_and_keep_checkpoint(self):
        env = FakeEnv([[(0, -4)], [(0, -4)], [(0, -4)]])
        env.game_window.refresh_on_engine_timeout = False
        def recover(status):
            self.assertTrue(env.game_window.refresh_on_engine_timeout)
            self.assertEqual(status, -4)
        env.game_window.recover.side_effect = recover
        release = Mock()
        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, "3 consecutive"):
            run_training(env, self.trainer(), TrainingState(), self.root, steps=8,
                         env_config=self.env_config, release_controls=release)
        self.assertTrue((self.root / "last.pt").is_file())
        self.assertFalse(env.game_window.refresh_on_engine_timeout)
        release.assert_called()

    def test_bc_initialization_cli_resume_and_bad_config(self):
        bc = self.root / "bc.pt"
        model = ActorCritic(hidden_sizes=(16,))
        torch.save({
            "format_version": 1, "training_stage": "behavioral_cloning",
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_names": list(ACTION_NAMES), "model_config": model.model_config,
            "model_state_dict": model.state_dict(),
        }, bc)
        parser = build_parser()
        output = self.root / "new"
        trainer, state, environment, _ = prepare_training(parser.parse_args([
            "--init", str(bc), "--output", str(output), "--device", "cpu",
        ]))
        self.assertEqual(trainer.model.model_config["hidden_sizes"], [16])
        self.assertFalse(trainer.optimizer.state)
        self.assertEqual(environment["combat_timeout"], 60)
        # 旧存档的时限不能覆盖当前 CLI 的默认 60 秒。
        environment["combat_timeout"] = 120
        save_checkpoint(output / "last.pt", trainer, state, environment)
        restored, _, resumed_environment, directory = prepare_training(parser.parse_args([
            "--resume", str(output / "last.pt"), "--device", "cpu", "--learning-rate", "0.002",
        ]))
        self.assertEqual(restored.config.learning_rate, 0.002)
        self.assertEqual(resumed_environment["combat_timeout"], 60)
        self.assertEqual(directory, output)
        _, _, custom_environment, _ = prepare_training(parser.parse_args([
            "--resume", str(output / "last.pt"), "--device", "cpu", "--combat-timeout", "45",
        ]))
        self.assertEqual(custom_environment["combat_timeout"], 45)
        with self.assertRaisesRegex(ValueError, "PPO checkpoint"):
            restore_checkpoint(bc)
        with self.assertRaises(ValueError):
            PPOConfig(gamma=2).validate()
        with self.assertRaises(ValueError):
            PPOConfig(learning_rate=float("nan")).validate()

    def test_legacy_or_wrong_schema_checkpoint_is_rejected(self):
        for dimension, version in ((60, None), (OBSERVATION_DIM, 1)):
            model = ActorCritic(observation_dim=dimension, hidden_sizes=(16,))
            path = self.root / "incompatible.pt"
            torch.save({
                "format_version": 1, "training_stage": "ppo", "action_names": list(ACTION_NAMES),
                "model_config": model.model_config, "model_state_dict": model.state_dict(),
                "observation_schema_version": version,
            }, path)
            with self.assertRaisesRegex(ValueError, "Incompatible observation schema"):
                load_policy(path)
            with self.assertRaisesRegex(ValueError, "Incompatible observation schema"):
                restore_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
