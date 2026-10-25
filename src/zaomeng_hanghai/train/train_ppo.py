"""在线 PPO：python -m zaomeng_hanghai.train.train_ppo；运行后控制游戏窗口。"""

import argparse
from collections import deque
from dataclasses import asdict, fields
from datetime import datetime
import json
import math
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from ..policy import ActorCritic, load_policy
from ..core.observation import OBSERVATION_SCHEMA_VERSION
from .ppo import PPOConfig, PPOTrainer, TrainingState, Transition, restore_checkpoint, save_checkpoint


def append_json(path, record):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


class RewardChangeLogger:
    """忽略 combat_time 分项，仅在其余奖励变化时打印、写入完整奖励日志。"""

    def __init__(self, state, path):
        self.state = state
        self.path = path
        # 中断前可能已打印一些尚未用于更新的奖励；同目录续训仍与真实末条日志比较。
        if path.exists():
            with path.open(encoding="utf-8") as file:
                last_line = deque((line for line in file if line.strip()), maxlen=1)
            if last_line:
                record = json.loads(last_line[0])
                self.state.last_logged_reward = self._comparison_reward(record["reward"], record)

    @staticmethod
    def _comparison_reward(reward, info):
        return float(reward) - float(info.get("reward_terms", {}).get("combat_time", 0.0))

    def log(self, reward, action, info):
        reward = float(reward)
        comparison_reward = self._comparison_reward(reward, info)
        if self.state.last_logged_reward is not None and math.isclose(
            self.state.last_logged_reward, comparison_reward, rel_tol=0.0, abs_tol=1e-12
        ):
            return False
        self.state.last_logged_reward = comparison_reward
        terms = {key: value for key, value in info.get("reward_terms", {}).items() if value != 0}
        detail = " ".join(f"{key}={value:+.6g}" for key, value in terms.items())
        print(f"[Reward] step={self.state.env_steps} episode={self.state.episodes} "
              f"action={action} reward={reward:+.6g} {detail}", flush=True)
        append_json(self.path, {
            "step": self.state.env_steps, "episode": self.state.episodes,
            "action": action, "reward": reward, "reward_terms": info.get("reward_terms", {}),
            "status_code": info.get("status_code", 0),
        })
        return True


def observation_tensor(observation, trainer):
    observation = np.asarray(observation, dtype=np.float32)
    expected = (trainer.model.model_config["observation_dim"],)
    if observation.shape != expected or not np.isfinite(observation).all():
        raise ValueError(f"Expected a finite observation with shape {expected}")
    return torch.as_tensor(observation, device=trainer.device)


def run_training(env, trainer, state, output, *, steps, env_config, max_engine_failures=3, release_controls=lambda: None):
    """整局采样、局间更新。支持注入模拟环境，测试时不操作游戏。"""
    target_steps = state.env_steps + steps
    logger = RewardChangeLogger(state, output / "rewards.jsonl")
    buffer: list[Transition] = []
    pending_recovery = None
    consecutive_failures = 0
    trainer.model.eval()
    # 稳定 checkpoint 只在完整 PPO 更新后替换；中断不会写入半次更新。
    save_checkpoint(output / "last.pt", trainer, state, env_config)
    print(f"Training: valid steps {state.env_steps} -> {target_steps}; output={output.resolve()}", flush=True)
    previous_refresh = getattr(env.game_window, "refresh_on_engine_timeout", False)
    env.game_window.refresh_on_engine_timeout = True
    try:
        while state.env_steps < target_steps:
            if pending_recovery is not None:
                env.game_window.recover(pending_recovery)
            state.episodes += 1
            observation, status = env.reset()
            episode_steps, episode_return = 0, 0.0
            if status == 0:
                while True:
                    with torch.no_grad():
                        action_tensor, log_prob, value = trainer.model.act(observation_tensor(observation, trainer))
                    action = int(action_tensor.item())
                    next_observation, reward, terminated, truncated, info = env.step_with_reward(action)
                    status = info["status_code"]
                    if status == -4 or not info.get("transition_valid", True):
                        # 丢弃故障步；上一个有效步骤可 bootstrap，但优势不能跨入下一局。
                        if buffer:
                            buffer[-1].trace_end = True
                        break
                    reward = float(reward)
                    if not np.isfinite(reward):
                        raise ValueError("Environment returned a non-finite reward")
                    with torch.no_grad():
                        # 真终止价值为 0；-3 时间截断使用截断前观测 bootstrap。
                        next_value = 0.0 if terminated else trainer.model.critic(
                            observation_tensor(next_observation, trainer)
                        ).squeeze(-1).item()
                    buffer.append(Transition(
                        np.asarray(observation, dtype=np.float32).copy(), action,
                        log_prob.item(), value.item(), reward, next_value, bool(terminated or truncated),
                    ))
                    state.env_steps += 1
                    episode_steps += 1
                    episode_return += reward
                    logger.log(reward, action, info)
                    observation = next_observation
                    if terminated or truncated:
                        break
            release_controls()
            pending_recovery = status
            valid_episode = episode_steps > 0 and status in (1, -2, -3)
            if valid_episode:
                consecutive_failures = 0
                state.successes += int(status == 1)
                state.recent_returns = (state.recent_returns + [episode_return])[-20:]
            else:
                consecutive_failures += 1
                state.engine_failures += int(status == -4)
                if buffer:
                    buffer[-1].trace_end = True
                print(f"[Engine] episode={state.episodes} status={status}; invalid step discarded", flush=True)
            append_json(output / "episodes.jsonl", {
                "episode": state.episodes, "step": state.env_steps, "status_code": status,
                "valid_episode": valid_episode, "valid_steps": episode_steps,
                "return": episode_return,
            })
            print(f"[Episode] {state.episodes} status={status} steps={episode_steps} "
                  f"return={episode_return:+.6g}", flush=True)
            if buffer and (len(buffer) >= trainer.config.rollout_steps or state.env_steps >= target_steps):
                if status in (-3, -4):
                    # 截断时游戏可能仍在战斗，先退出到新一局的起始区域再更新。
                    env.game_window.recover(status)
                    pending_recovery = None
                metrics = trainer.update(buffer)
                state.trained_steps += len(buffer)
                state.updates += 1
                buffer.clear()
                mean_return = float(np.mean(state.recent_returns)) if state.recent_returns else None
                improved = mean_return is not None and (
                    state.best_mean_return is None or mean_return > state.best_mean_return
                )
                if improved:
                    state.best_mean_return = mean_return
                save_checkpoint(output / "last.pt", trainer, state, env_config)
                if improved:
                    save_checkpoint(output / "best.pt", trainer, state, env_config)
                append_json(output / "updates.jsonl", {
                    "update": state.updates, "step": state.env_steps, "trained_steps": state.trained_steps,
                    "mean_return_20": mean_return, **metrics,
                })
                print(f"[PPO] update={state.updates} steps={state.env_steps} samples={metrics['samples']} "
                      f"policy_loss={metrics['policy_loss']:.5f} value_loss={metrics['value_loss']:.5f} "
                      f"entropy={metrics['entropy']:.4f} kl={metrics['approx_kl']:.5f}", flush=True)
            if consecutive_failures >= max_engine_failures:
                raise RuntimeError(f"{consecutive_failures} consecutive engine/reset failures; check the game window")
    except KeyboardInterrupt:
        print("Interrupted. Resume from last.pt; incomplete collection/update is discarded.", flush=True)
    finally:
        env.game_window.refresh_on_engine_timeout = previous_refresh
        release_controls()
    print(f"Resume checkpoint: {(output / 'last.pt').resolve()}", flush=True)
    return state


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--init", type=Path, help="从 BC 等 checkpoint 初始化权重，创建新的 PPO 优化器")
    source.add_argument("--resume", type=Path, help="恢复 PPO 权重、优化器、参数和进度")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=100_000, help="本次额外采集的有效步数，完成当前局后停止")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", help="新建模型默认 256 256 256；checkpoint 使用保存的结构")
    for name in ("rollout_steps", "update_epochs", "batch_size"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "entropy_coef", "value_coef", "max_grad_norm", "target_kl"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None)
    parser.add_argument("--combat-timeout", type=float, help="每关战斗时限，新训练默认 120 秒；续训沿用存档")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="新训练随机种子；续训恢复存档中的随机状态")
    parser.add_argument("--max-engine-failures", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    return parser


def prepare_training(args):
    if min(args.steps, args.threads, args.max_engine_failures) <= 0:
        raise ValueError("steps, threads and max-engine-failures must be positive")
    if args.hidden_sizes and min(args.hidden_sizes) <= 0:
        raise ValueError("Hidden widths must be positive")
    if args.hidden_sizes and (args.init or args.resume):
        raise ValueError("--hidden-sizes only applies to new models without --init/--resume")
    if args.combat_timeout is not None and (not np.isfinite(args.combat_timeout) or args.combat_timeout <= 0):
        raise ValueError("combat-timeout must be positive and finite")
    torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    overrides = {item.name: getattr(args, item.name) for item in fields(PPOConfig) if getattr(args, item.name) is not None}
    if args.resume:
        trainer, state, env_config = restore_checkpoint(args.resume, device, overrides)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = load_policy(args.init, device) if args.init else ActorCritic(hidden_sizes=args.hidden_sizes or (256, 256, 256)).to(device)
        trainer = PPOTrainer(model, PPOConfig(**overrides))
        state = TrainingState()
        env_config = {"timeout": 5, "stay_timeout": 30, "combat_timeout": 120}
    if args.combat_timeout is not None:
        env_config["combat_timeout"] = args.combat_timeout
    output = args.output or (args.resume.resolve().parent if args.resume else Path("runs/ppo") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    ))
    existing_resume = args.resume is not None and output.resolve() == args.resume.resolve().parent
    output.mkdir(parents=True, exist_ok=existing_resume)
    configuration = {
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "ppo": asdict(trainer.config), "environment": env_config,
        "model": trainer.model.model_config, "device": str(trainer.device),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if not existing_resume:
        (output / "config.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    append_json(output / "sessions.jsonl", configuration)
    return trainer, state, env_config, output


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        trainer, state, env_config, output = prepare_training(args)
    except ValueError as error:
        parser.error(str(error))
    # --help、算法测试和 checkpoint 恢复不导入/操作游戏窗口。
    from ..env import Env
    from ..core.action_manager import key_up_all, key_up

    def release_controls():
        key_up_all()
        key_up("J")

    env = Env(render_enabled=args.render, **env_config)
    try:
        env.init_window()
        run_training(env, trainer, state, output, steps=args.steps, env_config=env_config,
                     max_engine_failures=args.max_engine_failures, release_controls=release_controls)
    finally:
        release_controls()


if __name__ == "__main__":
    main()
