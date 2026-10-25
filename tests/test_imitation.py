"""离线验证数据过滤、整局划分、BC 优化和模型往返，不连接游戏。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from zaomeng_hanghai.train.expert_data import load_episodes, split_episodes
from zaomeng_hanghai.policy import ACTION_NAMES, ActorCritic, load_policy
from zaomeng_hanghai.train.train_bc import main, run_epoch


class ImitationTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "runs"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_bc_", dir=scratch)
        if not Path(self.temporary.name).resolve().is_relative_to(scratch.resolve()):
            raise ValueError("Test temporary directory must stay inside workspace runs/")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "config.json").write_text(json.dumps({
            "schema_version": 1, "observation_dim": 60, "actions": list(ACTION_NAMES)
        }), encoding="utf-8")
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def add_episode(self, index, kept=True, status=1, terminal_level=4, actions=None):
        observations = np.zeros((12, 60), dtype=np.float32)
        observations[:, 0] = index / 10
        actions = np.full(12, 3, dtype=np.int64) if actions is None else actions
        name = f"episode_{index:04d}.npz"
        if kept:
            np.savez(self.data / name, obs=observations, acts=actions)
        with (self.data / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps({
                "kept": kept, "status": status, "terminal_level": terminal_level,
                "sample_count": 12, "file": name,
            }) + "\n")

    def test_manifest_filter_and_episode_split(self):
        for index in range(1, 6):
            self.add_episode(index)
        self.add_episode(6, kept=False)
        self.add_episode(7, status=-2, terminal_level=1)
        self.add_episode(8, status=-4)
        # 不在 manifest 的 NPZ 绝不加载。
        np.savez(self.data / "orphan.npz", obs=np.zeros((1, 60)), acts=[0])
        episodes = load_episodes(self.data)
        self.assertEqual(len(episodes), 5)
        train, validation = split_episodes(episodes, 0.2, seed=8)
        self.assertEqual((len(train), len(validation)), (4, 1))
        self.assertFalse({e.path for e in train} & {e.path for e in validation})
        self.assertEqual([e.path for e in train], [e.path for e in split_episodes(episodes, 0.2, 8)[0]])

    def test_invalid_action_and_single_episode_validation(self):
        self.add_episode(1, actions=np.full(12, 9, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "0..8"):
            load_episodes(self.data)
        np.savez(self.data / "episode_0001.npz", obs=np.zeros((12, 60)), acts=np.zeros(12, dtype=np.int64))
        episodes = load_episodes(self.data)
        with self.assertRaisesRegex(ValueError, "at least two"):
            split_episodes(episodes)
        self.assertEqual(len(split_episodes(episodes, 0)[0]), 1)
        np.savez(self.data / "episode_0001.npz", obs=np.full((12, 60), np.nan), acts=np.zeros(12, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "finite"):
            load_episodes(self.data)

    def test_bc_learns_and_leaves_critic_unchanged(self):
        model = ActorCritic(hidden_sizes=(16,))
        critic_before = {key: value.clone() for key, value in model.critic.state_dict().items()}
        obs = torch.zeros(32, 60)
        obs[16:, 0] = 1
        actions = torch.cat((torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)))
        loader = [(obs, actions)]
        optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.03)
        before = run_epoch(model, loader, "cpu")
        for _ in range(70):
            run_epoch(model, loader, "cpu", optimizer)
        after = run_epoch(model, loader, "cpu")
        self.assertLess(after["loss"], before["loss"] * 0.1)
        self.assertEqual(after["accuracy"], 1)
        for key, value in model.critic.state_dict().items():
            torch.testing.assert_close(value, critic_before[key], rtol=0, atol=0)
        sampled, log_prob, values = model.act(obs)
        evaluated, entropy, evaluated_values = model.evaluate_actions(obs, sampled)
        torch.testing.assert_close(log_prob, evaluated)
        torch.testing.assert_close(values, evaluated_values)
        self.assertEqual(entropy.shape, (32,))

    def test_cli_and_checkpoint_roundtrip(self):
        self.add_episode(1)
        self.add_episode(2)
        output = self.root / "training"
        main(["--data", str(self.data), "--output", str(output), "--epochs", "2",
              "--hidden-sizes", "16", "--batch-size", "8", "--device", "cpu"])
        for name in ("best.pt", "last.pt", "config.json", "split.json", "history.jsonl"):
            self.assertTrue((output / name).is_file())
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        self.assertFalse(checkpoint["critic_trained"])
        model = load_policy(output / "best.pt")
        obs = np.zeros(60, dtype=np.float32)
        expected = ActorCritic(**checkpoint["model_config"])
        expected.load_state_dict(checkpoint["model_state_dict"])
        self.assertEqual(model.predict(obs), expected.predict(obs))
        with self.assertRaises(ValueError):
            model.predict(np.zeros(59))
        with self.assertRaises(FileExistsError):
            main(["--data", str(self.data), "--output", str(output), "--epochs", "1"])


if __name__ == "__main__":
    unittest.main()
