"""游戏环境流程编排：感知更新、动作调度、奖励和调试渲染。"""

import time

import cv2
import numpy as np

from zaomeng_hanghai.game_window import GameWindow
from zaomeng_hanghai.core.action_manager import ActionManager, click, key_down
from zaomeng_hanghai.core.constants import MAZE_HEIGHT, MAZE_WIDTH
from zaomeng_hanghai.core.direction import Direction
from zaomeng_hanghai.core.entities import Bullets, EnemyList, FallingItems, Player
from zaomeng_hanghai.core.map import Map
from zaomeng_hanghai.core.observation import (
    BULLET_SLICE, COLLISION_SLICE, ENEMY_SLICE, OBSERVATION_DIM, OBSERVATION_SCHEMA_VERSION,
)
from zaomeng_hanghai.core.rewards import RewardConfig, calculate_reward


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapse_time = time.time() - start_time
        if elapse_time > 0.02:
            print(f"Function '{func.__name__}' of Env executed in {elapse_time:.4f} seconds.")
        return result

    return wrapper


class FrameRateCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = None
        self.fps = 0.0
        self.report_start = None
        self.report_count = 0

    def trigger(self, log_to_terminal=False):
        """每次有效环境更新计一帧，终端每十秒报告一次平均更新频率。"""
        current_time = time.perf_counter()
        if self.last_time is not None:
            time_difference = current_time - self.last_time
            if time_difference > 0:
                self.fps = 1 / time_difference
            self.report_count += 1
        if self.report_start is None:
            self.report_start = current_time
        elapsed = current_time - self.report_start
        if elapsed >= 10.0:
            if log_to_terminal:
                average_fps = self.report_count / elapsed
                print(f"[FPS] {average_fps:.1f}，平均更新间隔 {elapsed * 1000 / self.report_count:.1f} ms（最近 {elapsed:.1f}s）")
            self.report_start = current_time
            self.report_count = 0
        self.last_time = current_time
        return self.fps


class Env:
    # 角色/关卡 17 + 五个敌人 40 + 三颗子弹 21 + 碰撞方向 8。
    state_space = OBSERVATION_DIM
    state_schema_version = OBSERVATION_SCHEMA_VERSION

    def __init__(self, timeout=5, stay_timeout=30, render_enabled=False, reward_config=None, combat_timeout=120):
        if not np.isfinite(combat_timeout) or combat_timeout <= 0:
            raise ValueError("combat_timeout must be positive and finite")
        self.render_enabled = render_enabled
        self.timeout = timeout
        self.game_window = GameWindow()
        self.window_size = None
        self.action_manager = ActionManager()
        self.action_space = 9

        self.player: Player | None = None
        self.enemy_list = EnemyList()
        self.bullets = Bullets()
        self.falling_items = FallingItems()
        self.new_level = False

        self.move_index = None
        self.move_end = None

        self.fps_calculator = FrameRateCalculator()
        self.map = Map(render_enabled=render_enabled)
        self.stay_timeout = stay_timeout
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.last_reward = 0.0
        self.reward_terms = {}
        self.last_step_info = {}
        self._episode_done = False
        self.combat_timeout = combat_timeout
        self._reset_combat_timer()

    def _reset_combat_timer(self):
        self._combat_level = None
        self._combat_elapsed = 0.0
        self._combat_started_at = None

    @property
    def combat_elapsed_seconds(self):
        elapsed = self._combat_elapsed
        if self._combat_started_at is not None:
            elapsed += max(0.0, time.monotonic() - self._combat_started_at)
        return elapsed

    def _pause_combat_timer(self):
        self._combat_elapsed = self.combat_elapsed_seconds
        self._combat_started_at = None

    def _update_combat_timer(self):
        """同关累计战斗时间；空场暂停，切关重置，加载时间不计入。"""
        if self._combat_level != self.map.level:
            self._reset_combat_timer()
            self._combat_level = self.map.level
        if not 1 <= self.map.level <= 4 or self.can_next_level():
            self._pause_combat_timer()
            return 0
        if self._combat_started_at is None:
            self._combat_started_at = time.monotonic()
        if self.combat_elapsed_seconds >= self.combat_timeout:
            self._pause_combat_timer()
            return -3
        return 0

    def init_window(self, outer_window_box=None, *, focus_game=True):
        self.game_window.init_window(outer_window_box)
        self.window_size = self.game_window.window_size

        if self.render_enabled:
            image = np.zeros((MAZE_HEIGHT, MAZE_WIDTH, 3), dtype=np.uint8)
            cv2.imshow("Monitor", image)
            cv2.moveWindow("Monitor", 0, 0)
            cv2.setWindowProperty("Monitor", cv2.WND_PROP_TOPMOST, 1)

        if focus_game:
            x1, y1, x2, y2 = self.game_window.window_box
            click(x2 - 10, y2 - 10)

    def reset(self):
        self.last_reward = 0.0
        self.reward_terms = {}
        self.last_step_info = {}
        self._episode_done = False
        self._reset_combat_timer()
        self.player = None
        self.action_manager.reset()
        self.fps_calculator.reset()
        self.enemy_list = EnemyList()
        self.bullets = Bullets()
        self.falling_items = FallingItems()

        self.new_level = False
        self.move_index = None
        self.move_end = None

        status_code = self.update()
        if status_code:
            self._episode_done = True
            return np.zeros(self.state_space, dtype=np.float32), status_code
        self.render()

        map_index = self.game_window.get_map_index_at_start()
        self.map.reset(map_index)
        self._reset_combat_timer()

        status_code = self.try_next_level()
        self._episode_done = status_code != 0
        state = self.state_encode()
        print("游戏开始！大地图：", map_index)

        return state, status_code

    def update(self) -> int:
        """更新感知、实体跟踪及战斗状态，供智能体和自动寻路共用。"""
        status_code = self.game_window.update()
        start_time = time.monotonic()
        if status_code == -1:
            self._pause_combat_timer()
        while status_code == -1:
            time.sleep(0.1)
            status_code = self.game_window.update()

            if status_code == -1 and time.monotonic() - start_time >= self.timeout:
                return -4

        # 结算画面没有新的实体检测结果，保留最后一次有效的场景状态。
        if status_code:
            self._pause_combat_timer()
            return status_code

        maze = self.map.get_maze()

        player = Player(self.game_window.player_box, maze)
        if self.player and self.player.is_next(player):
            self.map.next_level()
            self.new_level = True

            print(f"进入第 {self.map.level} 关：关卡类型 {self.map.get_task()}")

            # 下一关可能使用不同的地图
            maze = self.map.get_maze()
            player = Player(self.game_window.player_box, maze)

            # 清除上一关的追踪记录，避免跨关误匹配
            self.enemy_list.clear()
            self.bullets.clear()
            self.falling_items.clear()
            self.move_index = None
            self.move_end = None

        self.player = player

        self.bullets.update(self.game_window.bullet_list_normal, self.player)
        # 子弹列表已经在 update 内部按距离排序

        current_time = time.time()
        self.enemy_list.update(self.game_window.enemy_boxes, maze, current_time=current_time)
        self.enemy_list.sort_by(self.player)

        self.action_manager.update_pao(self.game_window.items["pao"])

        self.falling_items.update(self.game_window.falling_items_normal)
        self.falling_items.sort_by(self.player)

        for bullet in self.bullets.new_bullets:
            min_distance = np.inf
            fired_enemy = None
            for enemy in self.enemy_list:
                if bullet.d == 0 and enemy.y1 < bullet.y < enemy.y2:
                    distance = bullet.x - enemy.x
                elif bullet.d == 1 and enemy.x1 < bullet.x < enemy.x2:
                    distance = bullet.y - enemy.y
                elif bullet.d == 2 and enemy.y1 < bullet.y < enemy.y2:
                    distance = enemy.x - bullet.x
                elif bullet.d == 3 and enemy.x1 < bullet.x < enemy.x2:
                    distance = enemy.y - bullet.y
                else:
                    continue
                if distance < 0:
                    continue

                if distance < min_distance:
                    min_distance = distance
                    fired_enemy = enemy
            if fired_enemy is not None:
                fired_enemy.set_fired(current_time)

        # 使动作管理器同步时间
        self.action_manager.set_action_start_time()

        # 准备fire动作
        fire_direction = self.map.fire_direction(self.player, self.enemy_list)
        self.action_manager.set_fire_direction(fire_direction)

        # 观测更新后旧危险图失效；仅自动战斗寻路会重建它。
        self.map.flee_mazes = None
        self.map.debug_image = None

        self.fps_calculator.trigger(log_to_terminal=not self.render_enabled)
        return self._update_combat_timer()

    def update_navigation(self):
        """按当前观测生成自动战斗寻路使用的危险图。"""
        self.map.update(self.player, self.bullets, self.enemy_list)

    def auto_step(self):
        status_code = self.move_to_enemy_and_update()  # 含有update

        if status_code:
            return status_code

        status_code = self.try_start_level()  # 含有update
        if status_code:
            return status_code
        status_code = self.try_next_level()  # 含有update

        return status_code

    def debug_step(self):
        """只更新感知、专家建议路径和 Monitor；不执行移动、开火或过关。"""
        status_code = self.update()
        if status_code not in (0, -3):
            return status_code
        self.update_navigation()
        _, path, self.move_index = self.map.get_shot_place_direction(
            self.player, self.move_index, self.move_end
        )
        self.move_end = path[-1] if path else None
        self.render(path)
        return status_code

    def step(self, action):
        """执行动作并计算奖励；为兼容推理调用，仍返回 status_code。

        奖励保存在 last_reward/reward_terms；训练可调用 step_with_reward。
        自动过关接管前截取开火、静止和撞墙分项，防止导航动作覆盖它们。
        """
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.action_space:
            raise ValueError(f"Invalid action: {action}")
        if self._episode_done:
            raise RuntimeError("Episode finished; call reset() before step()")
        started = time.monotonic()
        self.last_reward = 0.0
        self.reward_terms = {}
        self.last_step_info = {}
        level_before = self.map.level
        hp_before = self.game_window.hit_point
        position_before = self._reward_position()
        wall_collision = bool(self.player is not None and self.map.is_move_collision(self.player, action))
        self.action_manager.take_action(action)
        fire_reward = float(self.action_manager.fire_reward)
        self.render()
        hp_lost = 0.0
        stationary = False
        error = None
        try:
            status_code = self.update()
            if status_code == 0:
                hp_after_action = self.game_window.hit_point
                hp_lost = max(0, hp_before - hp_after_action)
                position_after = self._reward_position()
                if position_before is not None and position_after is not None and self.map.level == level_before:
                    displacement = np.hypot(
                        position_after[0] - position_before[0], position_after[1] - position_before[1]
                    )
                    stationary = displacement <= self.reward_config.stationary_threshold
                status_code = self.try_next_level()
                if status_code == 0:
                    hp_lost += max(0, hp_after_action - self.game_window.hit_point)
                self.new_level = False
        except TimeoutError as exc:
            # 引擎问题截断，不归因于智能体。
            status_code = -4
            error = str(exc)

        levels_passed = max(0, self.map.level - level_before) if level_before >= 1 else 0
        self.reward_terms = calculate_reward(
            self.reward_config, fire_reward=fire_reward, stationary=stationary,
            wall_collision=wall_collision, hp_lost=hp_lost,
            levels_passed=levels_passed, status=status_code,
        )
        self.last_reward = float(sum(self.reward_terms.values()))
        if status_code:
            self._pause_combat_timer()
        self.last_step_info = {
            "status_code": status_code,
            "reward_terms": dict(self.reward_terms),
            "elapsed_seconds": time.monotonic() - started,
            "combat_elapsed_seconds": self.combat_elapsed_seconds,
            "transition_valid": status_code != -4,
        }
        if error is not None:
            self.last_step_info["error"] = error
        self._episode_done = status_code != 0
        return status_code

    def _reward_position(self):
        if self.player is None:
            return None
        # 使用检测位置而非 find_nearest_passable_point 修正位置，避免撞墙被投影掩盖。
        return self.player.xn * MAZE_WIDTH, self.player.yn * MAZE_HEIGHT

    def step_with_reward(self, action):
        """返回 (obs, reward, terminated, truncated, info)，不额外执行动作。"""
        status_code = self.step(action)
        return (
            self.state_encode(), self.last_reward,
            status_code in (1, -2), status_code in (-3, -4), dict(self.last_step_info),
        )

    def try_start_level(self):
        def loop_act(direction):
            self.action_manager.take_action_direction(direction)
            self.render()

            time.sleep(0.035)
            return self.update()

        while self.new_level:
            if self.map.get_task() == 5:
                status_code = loop_act(None)
                if status_code:
                    return status_code
                if not self.enemy_list:
                    continue
                if self.map.map_index == 1:
                    if self.enemy_list[0].xn < 0.6:
                        continue
                elif self.map.map_index == 5:
                    if 0.4 < self.enemy_list[0].xn < 0.6:
                        continue
            elif self.map.map_index == 3:
                if self.map.level == 1:
                    direction = Direction.Down
                    self.action_manager.take_action_direction(direction)
                    time.sleep(1)
                    status_code = self.update()
                    if status_code:
                        return status_code
            elif self.map.map_index == 4:
                if self.map.level == 1:
                    status_code = loop_act(Direction.DownLeft)
                    if status_code:
                        return status_code
                    if self.player.xn > 0.75:
                        continue
                elif self.map.level == 2:
                    direction = Direction.Left
                    status_code = loop_act(direction)
                    if status_code:
                        return status_code
                    if self.player.xn > 0.3:
                        continue
            elif self.map.map_index == 5:
                if self.map.level == 2:
                    direction = Direction.Left
                    status_code = loop_act(direction)
                    if status_code:
                        return status_code
                    if self.player.xn > 0.3:
                        continue

            self.new_level = False
        return 0

    def can_next_level(self):
        return len(self.enemy_list) == 0 and len(self.bullets) == 0

    def try_next_level(self):
        """使用静态地图拾取掉落物、前往出口，不计算战斗危险图。"""
        loop_start_time = time.monotonic()
        while self.can_next_level() and not self.new_level:
            direction, path = self.map.get_next_direction(self.player, self.falling_items)
            self.render(path)
            self.action_manager.take_action_direction(direction)
            status_code = self.update()

            if status_code:
                return status_code
            if time.monotonic() - loop_start_time >= self.stay_timeout and self.can_next_level() and not self.new_level:
                return -4

        return 0

    def move_to_enemy_and_update(self):
        if self.can_next_level():
            return 0
        if self.new_level:
            return 0

        self.update_navigation()
        direction, path, self.move_index = self.map.get_shot_place_direction(
            self.player, self.move_index, self.move_end
        )
        if direction is None:
            self.action_manager.take_action_direction(None)
            self.render()
            status_code = self.update()
            return status_code

        self.move_end = path[-1]
        self.action_manager.take_action_direction(direction)
        self.render(path)

        status_code = self.update()

        return status_code

    def state_encode(self):
        """返回 v2 的 86 维 float32 观测；终局状态码单独返回。

        [0:7] 角色，[7] 关卡进度，[8:17] 类型 one-hot，
        [17:57] 五个敌人的相对位置/碰撞框/冷却，[57:78] 三颗危险弹幕，
        [78:86] 右、下、左、上、右下、左下、右上、左上的碰撞标志。
        """
        state = np.zeros(self.state_space, dtype=np.float32)
        state[0] = self.player.xn
        state[1] = self.player.yn
        state[2] = np.clip(self.action_manager.get_fire_cool_down() / 0.6, 0, 1)
        state[3] = np.clip(self.game_window.hit_point / 4, 0, 1)
        state[4:7] = [
            bool(self.game_window.items[name]) for name in ("pao", "dan", "fan")
        ]

        state[7] = self.map.level / 4
        task = self.map.get_task()
        if task == -1:
            pass  # 起始区域没有关卡任务类型，保留全零。
        elif 0 <= task < 9:
            state[8 + task] = 1
        else:
            raise ValueError(f"Invalid task type: {task}")

        state[ENEMY_SLICE] = self.enemy_list.state_encode(self.player)
        state[BULLET_SLICE] = self.bullets.state_encode(self.player)
        state[COLLISION_SLICE] = [self.map.is_move_collision(self.player, action) for action in range(1, 9)]

        return state

    def render(self, path=None):
        if not self.render_enabled:
            return

        image = np.zeros((MAZE_HEIGHT, MAZE_WIDTH, 3), dtype=np.uint8)
        if self.map.debug_image is not None:
            image += self.map.debug_image

        cv2.circle(image, (self.player.x, self.player.y), 4, (0, 255, 0), -1)
        cv2.circle(image, (self.player.x, self.player.y), 1, (0, 0, 255), -1)
        for enemy in self.enemy_list:
            cv2.circle(image, (enemy.x, enemy.y), 4, (0, 0, 255), -1)
            cv2.rectangle(image, (enemy.x1, enemy.y1), (enemy.x2, enemy.y2), (0, 0, 255), 2)
        for bullet in self.bullets:
            cv2.circle(image, (bullet.x, bullet.y), 2, (0, 255, 255), -1)
            direction_delta = ((5, 0), (0, 5), (-5, 0), (0, -5))
            dx, dy = direction_delta[bullet.d]
            cv2.line(image, (bullet.x - dx, bullet.y - dy), (bullet.x, bullet.y), (0, 255, 255), 1)
        for bullet in self.bullets.safe_bullets:
            cv2.circle(image, (bullet.x, bullet.y), 2, (0, 255, 0), -1)
            direction_delta = ((5, 0), (0, 5), (-5, 0), (0, -5))
            dx, dy = direction_delta[bullet.d]
            cv2.line(image, (bullet.x - dx, bullet.y - dy), (bullet.x, bullet.y), (0, 255, 0), 1)
        for item in self.falling_items:
            cv2.circle(image, (item.x, item.y), 4, (255, 255, 0), -1)

        # 绘制迷宫
        maze = self.map.get_maze()
        image[maze] = 128

        if path:
            for i in range(len(path) - 1):
                cv2.line(image, path[i], path[i + 1], (255, 0, 0), 1)
            cv2.circle(image, path[-1], 2, (255, 0, 0), -1)

        fps = self.fps_calculator.fps

        # 在左上角显示 HP
        cv2.putText(image, f"HP: {self.game_window.hit_point}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)
        # 在右上角显示地图序列d
        cv2.putText(image, f"Map: {self.game_window.map_index}", (250, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        # 在右上角显示关卡
        cv2.putText(image, f"Level: {self.map.level}", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 在关卡下方显示当前布局模板编号
        cv2.putText(image, f"Task: {self.map.get_task()}", (250, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        # 在右下角显示帧率
        cv2.putText(image, f"FPS: {fps:.1f}", (250, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 左下角显示拥有物品
        for i, value in enumerate(self.game_window.items.values()):
            if value:
                thickness = -1
            else:
                thickness = 1
            cv2.rectangle(image, (5 + i * 15, 180), (15 + i * 15, 190), (255, 255, 255), thickness)

        cv2.imshow("Monitor", image)
        cv2.waitKey(1)


def auto_loop(env: Env, target=3):
    # 传入已经初始化好窗口的Env实例
    success_count = 0

    print("Start Loop")
    while success_count < target:
        print(f"Success {success_count}/{target}")

        state, status_code = env.reset()
        while status_code == 0:
            status_code = env.auto_step()
            if status_code == 1:
                success_count += 1
        time.sleep(1)
        env.game_window.recover(status_code)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import os

    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_DIR)

    parser = argparse.ArgumentParser(description="自动战斗或只读调试窗口")
    parser.add_argument("target", nargs="?", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="只显示感知和建议路径，不发送鼠标或键盘指令")
    parser.add_argument("--map-index", type=int, choices=range(7), help="调试时指定大地图编号，默认从画面识别")
    parser.add_argument("--level", type=int, choices=range(5), default=0, help="调试开始时所在关卡，默认 0")
    args = parser.parse_args()

    env_ = Env(render_enabled=True)
    env_.init_window(focus_game=not args.debug)
    if args.debug:
        map_index = args.map_index if args.map_index is not None else env_.game_window.get_map_index_at_start()
        env_.map.reset(map_index)
        for _ in range(args.level):
            env_.map.next_level()
        print("只读调试：请手动操作游戏，Ctrl+C 退出。")
        try:
            while env_.debug_step() in (0, -3):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyWindow("Monitor")
    else:
        auto_loop(env_, args.target)
