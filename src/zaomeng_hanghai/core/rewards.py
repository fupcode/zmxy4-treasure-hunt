"""以成功开火 +1 为单位的奖励；不依赖游戏窗口或危险图。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    step: float = -0.02
    stationary: float = -0.005
    wall_collision: float = -0.2
    damage_per_hp: float = -5.0
    next_level: float = 5.0
    success: float = 30.0
    death: float = -20.0
    timeout: float = -10.0
    # 在 320×200 逻辑地图中的像素位移，过滤检测框轻微抖动。
    stationary_threshold: float = 1.0


def calculate_reward(
    config: RewardConfig, *, fire_reward: float, stationary: bool,
    wall_collision: bool, hp_lost: float, levels_passed: int, status: int,
) -> dict[str, float]:
    """返回可相加的奖励分项。撞墙时不再叠加静止惩罚。"""
    terms = {
        "fire": float(fire_reward),
        "step": config.step,
        "stationary": config.stationary if stationary and not wall_collision else 0.0,
        "wall_collision": config.wall_collision if wall_collision else 0.0,
        "damage": config.damage_per_hp * max(0.0, hp_lost),
        "next_level": config.next_level * max(0, levels_passed),
        "success": config.success if status == 1 else 0.0,
        "death": config.death if status == -2 else 0.0,
        "timeout": config.timeout if status == -3 else 0.0,
    }
    # 引擎故障的整步作废，连每步/静止/撞墙等惩罚也不归因给智能体。
    return dict.fromkeys(terms, 0.0) if status == -4 else terms
