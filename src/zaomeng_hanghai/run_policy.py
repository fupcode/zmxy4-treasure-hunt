"""使用 BC actor 控制游戏；执行此命令会操作当前游戏窗口。"""

import argparse
import time

from .policy import load_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-episode-seconds", type=float, default=300)
    parser.add_argument("--combat-timeout", type=float, default=30, help="同一关累计战斗时限（秒），默认 30")
    args = parser.parse_args(argv)
    if args.episodes <= 0 or not 0 < args.max_episode_seconds < float("inf"):
        parser.error("episodes and max-episode-seconds must be positive and finite")
    if not 0 < args.combat_timeout < float("inf"):
        parser.error("combat-timeout must be positive and finite")
    model = load_policy(args.checkpoint, args.device)
    # 离线训练与 --help 不导入窗口、检测器和按键控制依赖。
    from .env import Env
    from .core.action_manager import key_up_all, key_up

    env = Env(render_enabled=args.render, combat_timeout=args.combat_timeout)
    try:
        env.init_window()
        for episode in range(args.episodes):
            started = time.monotonic()
            state, status = env.reset()
            while status == 0:
                if time.monotonic() - started > args.max_episode_seconds:
                    # 外部运行预算属于执行截断，不能冒充单关战斗超时。
                    status = -4
                    print("Episode wall-clock budget exhausted", flush=True)
                    break
                action = model.predict(state, deterministic=not args.stochastic)
                status = env.step(action)
                if status == 0:
                    state = env.state_encode()
            key_up_all()
            print(f"Episode {episode + 1}: status={status}", flush=True)
            if episode + 1 < args.episodes:
                time.sleep(1)
                env.game_window.recover(status)
    finally:
        key_up_all()
        key_up("J")


if __name__ == "__main__":
    main()
