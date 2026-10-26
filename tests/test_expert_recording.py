"""离线验证专家自动采集、动作入口快照和 BC 格式兼容。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from zaomeng_hanghai.core.direction import Direction, direction_list
from zaomeng_hanghai.core.expert_recorder import ExpertRecorder
from zaomeng_hanghai.core.observation import OBSERVATION_DIM
from zaomeng_hanghai.env import Env, auto_loop
from zaomeng_hanghai.train.expert_data import load_episodes


class OptionalRecordingTests(unittest.TestCase):
    def make_env(self):
        env = Env.__new__(Env)
        env.map = SimpleNamespace(level=1)
        env.combat_timeout = 30
        env.reset = Mock(return_value=(None, 0))
        env.auto_step = Mock(return_value=1)
        env.game_window = SimpleNamespace(recover=Mock())
        env.state_encode = Mock(side_effect=AssertionError('Default mode must not record'))
        return env

    def test_default_single_run_does_not_create_recorder(self):
        env = self.make_env()
        with patch('zaomeng_hanghai.env.ExpertRecorder', side_effect=AssertionError('Unexpected recorder')):
            self.assertEqual(env.auto_run(), 1)
        self.assertIsNone(env.expert_data_path)
        self.assertIsNone(env._expert_recorder)
        env.state_encode.assert_not_called()

    def test_default_loop_runs_without_recording(self):
        env = self.make_env()
        with patch('zaomeng_hanghai.env.ExpertRecorder', side_effect=AssertionError('Unexpected recorder')), \
             patch('zaomeng_hanghai.env.time.sleep'):
            auto_loop(env, target=2)
        self.assertEqual(env.auto_step.call_count, 2)
        self.assertEqual(env.game_window.recover.call_count, 2)
        env.state_encode.assert_not_called()

    def test_explicit_flag_creates_recorder(self):
        env = self.make_env()
        with patch('zaomeng_hanghai.env.ExpertRecorder') as factory:
            env.auto_run(collect_expert=True)
        factory.assert_called_once_with(None, combat_timeout=30)
        factory.return_value.begin.assert_called_once_with()
        factory.return_value.finish.assert_called_once_with(1, 1)

    def test_interrupt_without_recording_preserves_exception(self):
        env = self.make_env()
        env.auto_step.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            env.auto_run()
        self.assertIsNone(env._expert_recorder)


class ExpertRecordingTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "runs"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_expert_", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "collection"
        self.env = Env.__new__(Env)
        self.env.combat_timeout = 120
        self.env.map = SimpleNamespace(level=1, map_index=2, get_task=lambda: 3)
        self.obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        self.env.state_encode = lambda: self.obs
        self.env.action_manager = SimpleNamespace(take_action_direction=Mock())
        self.env.reset = Mock(return_value=(self.obs, 0))

    def entries(self):
        return [json.loads(line) for line in (self.output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_auto_run_records_before_action_and_loads_in_bc(self):
        def action(direction):
            self.obs[0] = 0.9
        self.env.action_manager.take_action_direction.side_effect = action

        def step():
            self.env._take_expert_action(Direction.UpLeft, "combat")
            self.env.map.level = 4
            return 1

        self.env.auto_step = step
        self.assertEqual(self.env.auto_run(expert_output=self.output), 1)
        self.assertIsNone(self.env._expert_recorder)
        episode = load_episodes(self.output)[0]
        self.assertEqual(episode.observations[0, 0], 0)
        self.assertEqual(episode.actions.tolist(), [8])
        with np.load(episode.path, allow_pickle=False) as data:
            self.assertEqual(data["obs"].shape, (1, OBSERVATION_DIM))
            self.assertEqual(data["phases"].tolist(), [1])
            self.assertEqual(data["levels"].tolist(), [1])
            self.assertEqual(data["acts"].dtype, np.dtype("int64"))
        self.assertEqual(json.loads((self.output / "progress.json").read_text())["state"], "completed")

    def test_navigation_ends_dt_and_breaks_segment(self):
        with patch("zaomeng_hanghai.core.expert_recorder.time.monotonic", return_value=10) as clock:
            recorder = ExpertRecorder(self.output, combat_timeout=120)
            recorder.begin()
            clock.return_value = 11
            recorder.action(self.env, None, "start")
            clock.return_value = 12
            recorder.action(self.env, Direction.Right, "navigation")
            clock.return_value = 15
            recorder.action(self.env, Direction.Down, "combat")
            clock.return_value = 17
            recorder.finish(1, 4)
        with np.load(self.output / self.entries()[0]["file"], allow_pickle=False) as data:
            self.assertEqual(data["acts"].tolist(), [0, 2])
            self.assertEqual(data["segments"].tolist(), [0, 1])
            self.assertEqual(data["timestamps"].tolist(), [1, 5])
            self.assertEqual(data["dt"].tolist(), [1, 2])

    def test_action_mapping_and_filtering(self):
        recorder = ExpertRecorder(self.output, combat_timeout=120)
        for status, level, kept, reason in (
            (-2, 1, False, "died_at_level_1"), (-2, 2, True, None),
            (-3, 1, True, None), (-4, 3, False, "collection_error"),
        ):
            recorder.begin()
            self.env.map.level = level
            for direction in [None] + direction_list:
                recorder.action(self.env, direction, "combat")
            recorder.finish(status, level)
            entry = self.entries()[-1]
            self.assertEqual(entry["kept"], kept)
            self.assertEqual(entry["discard_reason"], reason)
            if kept:
                with np.load(self.output / entry["file"], allow_pickle=False) as data:
                    self.assertEqual(data["acts"].tolist(), list(range(9)))
        self.assertEqual(len(load_episodes(self.output)), 2)
        self.assertEqual(len(list(self.output.glob("*.npz"))), 2)

    def test_level_zero_and_no_samples_are_excluded(self):
        recorder = ExpertRecorder(self.output, combat_timeout=120)
        recorder.begin()
        self.env.map.level = 0
        self.env.state_encode = Mock(side_effect=AssertionError("Must not encode level zero"))
        recorder.action(self.env, Direction.Right, "combat")
        recorder.finish(1, 4)
        self.assertEqual(self.entries()[0]["discard_reason"], "no_samples")

    def test_interrupt_discards_current_episode_and_restores_action_path(self):
        def step():
            self.env._take_expert_action(None, "combat")
            raise KeyboardInterrupt()
        self.env.auto_step = step
        with self.assertRaises(KeyboardInterrupt):
            self.env.auto_run(expert_output=self.output)
        self.assertIsNone(self.env._expert_recorder)
        self.assertEqual(self.entries()[0]["status"], -4)
        self.assertFalse(list(self.output.glob("*.npz")))
        self.assertEqual(json.loads((self.output / "progress.json").read_text())["state"], "interrupted")
        self.env.state_encode = Mock(side_effect=AssertionError("Unexpected recording"))
        self.env._take_expert_action(None, "combat")

    def test_auto_loop_shares_run_directory_and_counts_successes(self):
        statuses = iter([-2, 1, 1])
        def step():
            self.env._take_expert_action(None, "combat")
            self.env.map.level = 2
            return next(statuses)
        self.env.auto_step = step
        self.env.game_window = SimpleNamespace(recover=Mock())
        with patch("zaomeng_hanghai.env.time.sleep"):
            auto_loop(self.env, 2, expert_output=self.output)
        self.assertEqual(len(self.entries()), 3)
        self.assertEqual(self.env.reset.call_count, 3)
        self.assertEqual(self.env.game_window.recover.call_count, 3)
        self.assertEqual(len(load_episodes(self.output)), 3)

    def test_existing_output_is_not_overwritten(self):
        ExpertRecorder(self.output, combat_timeout=120)
        with self.assertRaises(FileExistsError):
            ExpertRecorder(self.output, combat_timeout=120)


if __name__ == "__main__":
    unittest.main()
