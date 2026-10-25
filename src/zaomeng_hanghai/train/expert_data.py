"""只读取清单确认保留的专家局，不构造 reward 或相邻帧转移。"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from zaomeng_hanghai.policy import ACTION_NAMES, OBSERVATION_DIM


@dataclass
class ExpertEpisode:
    path: Path
    observations: np.ndarray
    actions: np.ndarray


def load_episodes(root: str | Path) -> list[ExpertEpisode]:
    root = Path(root).resolve()
    manifests = sorted(root.rglob("manifest.jsonl"))
    if not manifests:
        raise ValueError(f"No manifest.jsonl found under {root}")
    episodes = []
    seen = set()
    for manifest in manifests:
        config = json.loads((manifest.parent / "config.json").read_text(encoding="utf-8"))
        if (config.get("schema_version") != 1
                or config.get("observation_dim") != OBSERVATION_DIM
                or config.get("actions") != list(ACTION_NAMES)):
            raise ValueError(f"Unsupported expert schema: {manifest.parent}")
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if not entry["kept"]:
                    continue
                # 防御性地执行采集过滤规则，包括旧清单中的误标记。
                if entry["status"] == -4 or (entry["status"] == -2 and entry["terminal_level"] == 1):
                    continue
                path = (manifest.parent / entry["file"]).resolve()
                if not path.is_relative_to(manifest.parent) or path.suffix != ".npz":
                    raise ValueError("Episode must be an NPZ inside its run directory")
                if path in seen:
                    raise ValueError(f"Duplicate episode: {path}")
                with np.load(path, allow_pickle=False) as data:
                    obs, acts = data["obs"].copy(), data["acts"].copy()
                if obs.ndim != 2 or obs.shape[1] != OBSERVATION_DIM or len(obs) == 0:
                    raise ValueError(f"Expected nonempty (N, {OBSERVATION_DIM}) observations")
                if not np.issubdtype(obs.dtype, np.floating) or not np.isfinite(obs).all():
                    raise ValueError("Observations must be finite floats")
                if acts.shape != (len(obs),) or not np.issubdtype(acts.dtype, np.integer):
                    raise ValueError("Expected integer actions with shape (N,)")
                if np.any((acts < 0) | (acts >= len(ACTION_NAMES))):
                    raise ValueError("Actions must be in 0..8")
                if entry["sample_count"] != len(obs):
                    raise ValueError("Manifest sample_count differs from NPZ")
                episodes.append(ExpertEpisode(path, obs.astype(np.float32), acts.astype(np.int64)))
                seen.add(path)
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ValueError(f"{manifest}:{line_number}: {error}") from error
    if not episodes:
        raise ValueError(f"No retained expert episodes under {root}")
    return episodes


def split_episodes(episodes, validation_fraction=0.2, seed=42):
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction and len(episodes) < 2:
        raise ValueError("Validation requires at least two episodes; use --validation-fraction 0")
    indices = np.random.default_rng(seed).permutation(len(episodes))
    count = min(len(episodes) - 1, max(1, round(len(episodes) * validation_fraction))) if validation_fraction else 0
    return ([episodes[i] for i in indices[count:]], [episodes[i] for i in indices[:count]])


def concatenate(episodes):
    return (np.concatenate([episode.observations for episode in episodes]),
            np.concatenate([episode.actions for episode in episodes]))
