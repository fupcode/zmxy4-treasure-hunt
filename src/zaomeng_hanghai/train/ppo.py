"""离散动作 PPO、GAE 与可恢复 checkpoint；不导入游戏控制依赖。"""

from dataclasses import asdict, dataclass, field
import random

import numpy as np
import torch
from torch import nn

from ..policy import ACTION_NAMES, OBSERVATION_DIM, ActorCritic
from ..core.observation import OBSERVATION_SCHEMA_VERSION, validate_observation_schema


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    rollout_steps: int = 2048
    update_epochs: int = 10
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    def validate(self):
        if not all(np.isfinite(value) for value in asdict(self).values()):
            raise ValueError("PPO parameters must be finite")
        if min(self.rollout_steps, self.update_epochs, self.batch_size) <= 0:
            raise ValueError("rollout_steps, update_epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1]; gae_lambda in [0, 1]")
        if not 0 < self.clip_range < 1 or min(self.entropy_coef, self.value_coef, self.target_kl) < 0:
            raise ValueError("clip_range must be in (0, 1); loss coefficients and target_kl >= 0")


@dataclass
class TrainingState:
    env_steps: int = 0
    trained_steps: int = 0
    updates: int = 0
    episodes: int = 0
    successes: int = 0
    engine_failures: int = 0
    last_logged_reward: float | None = None
    recent_returns: list[float] = field(default_factory=list)
    best_mean_return: float | None = None


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    next_value: float
    trace_end: bool


def compute_gae(rewards, values, next_values, trace_ends, gamma, gae_lambda):
    """next_value 已处理终止 bootstrap；trace_end 阻止优势跨越 reset/故障。"""
    advantages = torch.zeros_like(rewards)
    following = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_values[index] - values[index]
        following = delta + gamma * gae_lambda * (~trace_ends[index]).float() * following
        advantages[index] = following
    return advantages, advantages + values


class PPOTrainer:
    def __init__(self, model: ActorCritic, config: PPOConfig):
        config.validate()
        if (model.model_config["observation_dim"] != OBSERVATION_DIM
                or model.model_config["action_dim"] != len(ACTION_NAMES)):
            raise ValueError(f"PPO expects the environment's {OBSERVATION_DIM} observations and 9 actions")
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def update(self, transitions: list[Transition]):
        if not transitions:
            raise ValueError("Cannot update PPO with an empty rollout")
        config = self.config
        obs = torch.as_tensor(np.stack([item.observation for item in transitions]), device=self.device)
        actions = torch.tensor([item.action for item in transitions], device=self.device)
        old_log_probs = torch.tensor([item.log_prob for item in transitions], device=self.device)
        values = torch.tensor([item.value for item in transitions], device=self.device)
        rewards = torch.tensor([item.reward for item in transitions], device=self.device)
        next_values = torch.tensor([item.next_value for item in transitions], device=self.device)
        trace_ends = torch.tensor([item.trace_end for item in transitions], device=self.device)
        advantages, returns = compute_gae(rewards, values, next_values, trace_ends, config.gamma, config.gae_lambda)
        # 保留原始优势尺度训练 critic；仅标准化 actor 的优势。
        if len(transitions) > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0, clip_fraction=0.0)
        count, epochs_completed, early_stop = 0, 0, False
        self.model.train()
        for epoch in range(config.update_epochs):
            indices = torch.randperm(len(transitions), device=self.device)
            for start in range(0, len(indices), config.batch_size):
                batch = indices[start:start + config.batch_size]
                log_probs, entropy, predicted_values = self.model.evaluate_actions(obs[batch], actions[batch])
                log_ratio = log_probs - old_log_probs[batch]
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                # 先检查 KL 再更新，避免继续放大已经过大的策略变化。
                if config.target_kl and approx_kl.item() > 1.5 * config.target_kl:
                    early_stop = True
                    break
                unclipped = ratio * advantages[batch]
                clipped = ratio.clamp(1 - config.clip_range, 1 + config.clip_range) * advantages[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * nn.functional.mse_loss(predicted_values, returns[batch])
                entropy_mean = entropy.mean()
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                batch_metrics = {
                    "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                    "entropy": entropy_mean.item(), "approx_kl": approx_kl.item(),
                    "clip_fraction": ((ratio - 1).abs() > config.clip_range).float().mean().item(),
                }
                for key, value in batch_metrics.items():
                    totals[key] += value * len(batch)
                count += len(batch)
            if early_stop:
                break
            epochs_completed = epoch + 1
        self.model.eval()
        return {**{key: value / max(count, 1) for key, value in totals.items()},
                "samples": len(transitions), "epochs_completed": epochs_completed, "kl_early_stop": early_stop}


def save_checkpoint(path, trainer: PPOTrainer, state: TrainingState, env_config):
    numpy_state = np.random.get_state()
    payload = {
        "format_version": 1, "training_stage": "ppo", "critic_trained": state.updates > 0,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_names": list(ACTION_NAMES), "model_config": trainer.model.model_config,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "ppo_config": asdict(trainer.config), "training_state": asdict(state), "env_config": env_config,
        "rng_state": {
            "torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, device="cpu", overrides=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format_version") != 1 or payload.get("training_stage") != "ppo"
            or payload.get("action_names") != list(ACTION_NAMES)):
        raise ValueError("--resume requires a PPO checkpoint; use --init for BC weights")
    validate_observation_schema(payload.get("observation_schema_version"),
                                payload["model_config"]["observation_dim"])
    config = PPOConfig(**{**payload["ppo_config"], **(overrides or {})})
    model = ActorCritic(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    trainer = PPOTrainer(model, config)
    trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    for group in trainer.optimizer.param_groups:
        group["lr"] = config.learning_rate
    state = TrainingState(**payload["training_state"])
    rng = payload["rng_state"]
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    name, keys, position, has_gauss, cached_gaussian = rng["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
    if torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(rng["cuda"])
    model.eval()
    return trainer, state, payload["env_config"]
