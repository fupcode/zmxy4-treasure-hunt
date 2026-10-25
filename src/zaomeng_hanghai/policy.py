"""PPO 可复用的离散动作 Actor-Critic；当前仅对 actor 做行为克隆。"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from .core.observation import OBSERVATION_DIM, OBSERVATION_SCHEMA_VERSION, validate_observation_schema


ACTION_NAMES = (
    "stop", "Right", "Down", "Left", "Up", "DownRight", "DownLeft", "UpRight", "UpLeft"
)


def mlp(input_dim, hidden_sizes, output_dim):
    layers = []
    for width in hidden_sizes:
        layers.extend((nn.Linear(input_dim, width), nn.Tanh()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """独立 actor/critic，BC 不修改尚未训练的 critic。"""

    def __init__(self, observation_dim=OBSERVATION_DIM, action_dim=9, hidden_sizes=(256, 256, 256)):
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0 or not hidden_sizes or min(hidden_sizes) <= 0:
            raise ValueError("Model dimensions must be positive")
        self.model_config = {
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "hidden_sizes": list(hidden_sizes),
        }
        self.actor = mlp(observation_dim, hidden_sizes, action_dim)
        self.critic = mlp(observation_dim, hidden_sizes, 1)

    def forward(self, observations):
        return self.actor(observations), self.critic(observations).squeeze(-1)

    def evaluate_actions(self, observations, actions):
        """为未来 PPO 更新提供 log_prob、entropy 和 value。"""
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), values

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        logits, values = self(observations)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), values

    @torch.no_grad()
    def predict(self, observation: np.ndarray, deterministic=True) -> int:
        device = next(self.parameters()).device
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.model_config["observation_dim"],) or not np.isfinite(observation).all():
            raise ValueError("Expected one finite observation with the model's observation dimension")
        action, _, _ = self.act(torch.as_tensor(observation, device=device), deterministic)
        return int(action.item())


def load_policy(path: str | Path, device="cpu") -> ActorCritic:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("action_names") != list(ACTION_NAMES):
        raise ValueError("Unsupported checkpoint format or action mapping")
    validate_observation_schema(checkpoint.get("observation_schema_version"),
                                checkpoint["model_config"]["observation_dim"])
    model = ActorCritic(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
