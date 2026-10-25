"""自动专家运行的 v2 行为克隆样本；不依赖游戏或训练框架。"""

from datetime import datetime
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from .direction import direction_list
from .observation import OBSERVATION_DIM, OBSERVATION_SCHEMA_VERSION


class ExpertRecorder:
    def __init__(self, output=None, *, combat_timeout):
        root = Path(__file__).resolve().parents[3] / "data" / "expert"
        self.path = Path(output) if output is not None else root / (
            datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:8]
        )
        self.path = self.path.resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.finished = self.kept = self.successful = self.saved_samples = 0
        self.active = None
        self._json("config.json", {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "observation_dim": OBSERVATION_DIM,
            "actions": ["stop"] + [direction.name for direction in direction_list],
            "phases": {"start": 0, "combat": 1},
            "combat_timeout": combat_timeout,
            "filters": ["died_at_level_1", "no_samples", "collection_error"],
        })
        (self.path / "manifest.jsonl").touch()
        self.progress("created")
        print(f"Expert data directory: {self.path}")

    def _json(self, name, value):
        temporary = self.path / (name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path / name)

    def progress(self, state, error=None):
        self._json("progress.json", {
            "state": state, "finished_episodes": self.finished,
            "active_episode": self.active, "kept_episodes": self.kept,
            "discarded_episodes": self.finished - self.kept,
            "successful_episodes": self.successful, "saved_samples": self.saved_samples,
            "error": error,
        })

    def begin(self):
        self.active = self.finished + 1
        self.started = time.monotonic()
        self.rows = []
        self.pending = None
        self.previous_key = None
        self.segment = -1
        self.progress("running")

    def _end_action(self, now):
        if self.pending is not None:
            row = self.rows[self.pending]
            row["dt"] = max(0.0, now - row["timestamps"])
            self.pending = None

    def action(self, env, direction, phase):
        now = time.monotonic() - self.started
        self._end_action(now)
        if phase not in ("start", "combat") or not 1 <= env.map.level <= 4:
            self.previous_key = None
            return
        obs = np.array(env.state_encode(), dtype=np.float32, copy=True)
        if obs.shape != (OBSERVATION_DIM,) or not np.isfinite(obs).all():
            raise ValueError("Invalid expert observation")
        key = (phase, env.map.level, env.map.map_index, env.map.get_task())
        if key != self.previous_key:
            self.segment += 1
        self.previous_key = key
        self.rows.append({
            "obs": obs, "acts": 0 if direction is None else direction_list.index(direction) + 1,
            "phases": 0 if phase == "start" else 1, "levels": env.map.level,
            "tasks": env.map.get_task(), "map_indices": env.map.map_index,
            "segments": self.segment, "timestamps": now, "dt": 0.0,
        })
        self.pending = len(self.rows) - 1

    def finish(self, status, terminal_level, error=None):
        duration = time.monotonic() - self.started
        self._end_action(duration)
        reason = None
        if status not in (1, -2, -3) or error is not None:
            reason = "collection_error"
        elif status == -2 and terminal_level == 1:
            reason = "died_at_level_1"
        elif not self.rows:
            reason = "no_samples"
        name = None
        if reason is None:
            dtypes = {"obs": np.float32, "acts": np.int64, "phases": np.int8,
                      "levels": np.int8, "tasks": np.int8, "map_indices": np.int8,
                      "segments": np.int32, "timestamps": np.float64, "dt": np.float32}
            arrays = {key: np.asarray([row[key] for row in self.rows], dtype=dtype)
                      for key, dtype in dtypes.items()}
            arrays.update(episode_id=np.int32(self.active), status=np.int8(status),
                          terminal_level=np.int8(terminal_level))
            name = f"episode_{self.active:04d}.npz"
            temporary = self.path / (name + ".tmp")
            with temporary.open("wb") as file:
                np.savez_compressed(file, **arrays)
            temporary.replace(self.path / name)
        entry = {
            "episode_id": self.active, "status": status, "success": status == 1,
            "terminal_level": terminal_level,
            "max_recorded_level": max((row["levels"] for row in self.rows), default=0),
            "kept": reason is None, "discard_reason": reason, "file": name,
            "sample_count": len(self.rows),
            "start_samples": sum(row["phases"] == 0 for row in self.rows),
            "combat_samples": sum(row["phases"] == 1 for row in self.rows),
            "duration_seconds": duration, "error": error,
        }
        with (self.path / "manifest.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        self.finished += 1
        self.kept += reason is None
        self.successful += status == 1
        self.saved_samples += len(self.rows) if reason is None else 0
        self.active = None
        self.rows = []
        self.progress("running")
        result = "已保存" if reason is None else f"已丢弃（{reason}）"
        print(
            f"[专家样本] 第 {entry['episode_id']} 局：状态 {status}，终止关卡 {terminal_level}，"
            f"耗时 {duration:.1f}s；采集 {entry['sample_count']} 对状态—动作样本"
            f"（开局 {entry['start_samples']}，战斗 {entry['combat_samples']}），{result}。"
        )
        print(
            f"[专家样本] 当前目录累计：保存 {self.saved_samples} 对，保留 {self.kept} 局，"
            f"丢弃 {self.finished - self.kept} 局；目录：{self.path}"
        )
