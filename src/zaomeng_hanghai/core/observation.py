"""环境、训练数据和模型共用的观测布局。修改字段语义时必须升级版本。"""

OBSERVATION_SCHEMA_VERSION = 2
BASE_DIM = 17
MAX_ENEMIES = 5
MAX_BULLETS = 3
ENEMY_FIELDS = ("present", "dx", "dy", "box_left_dx", "box_top_dy", "box_right_dx", "box_bottom_dy")
BULLET_FIELDS = ("present", "right", "down", "left", "up", "dx", "dy")
ENEMY_SLICE = slice(BASE_DIM, BASE_DIM + MAX_ENEMIES * len(ENEMY_FIELDS))
BULLET_SLICE = slice(ENEMY_SLICE.stop, ENEMY_SLICE.stop + MAX_BULLETS * len(BULLET_FIELDS))
COLLISION_SLICE = slice(BULLET_SLICE.stop, BULLET_SLICE.stop + 8)
OBSERVATION_DIM = COLLISION_SLICE.stop


def validate_observation_schema(version, dimension):
    if version != OBSERVATION_SCHEMA_VERSION or dimension != OBSERVATION_DIM:
        raise ValueError(
            f"Incompatible observation schema: expected version {OBSERVATION_SCHEMA_VERSION}, "
            f"{OBSERVATION_DIM} features; got version {version}, {dimension} features. "
            "Legacy 60-feature data/checkpoints cannot be reused directly; collect new expert data or retrain."
        )
