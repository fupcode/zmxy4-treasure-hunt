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
        self.last_time = None  # 上一次触发的时间戳
        self.fps = 0  # 当前帧率

    def trigger(self):
        current_time = time.time()  # 获取当前时间戳
        if self.last_time is not None:
            # 计算帧率（1 / 时间差）
            time_difference = current_time - self.last_time
            if time_difference > 0:
                self.fps = int(1 / time_difference)

        # 更新上一次触发时间
        self.last_time = current_time

        return self.fps


class Env:
    def __init__(self, timeout=5, stay_timeout=180, act_interval=0.02):
        self.timeout = timeout
        self.game_window = GameWindow()
        self.window_size = None
        self.action_manager = ActionManager()
        self.action_space = 9
        self.state_space = 7 + 10 + 3 * 6 + 7 * 4
        self.act_interval = act_interval

        self.player: Player | None = None
        self.enemy_list = EnemyList()
        self.bullets = Bullets()
        self.falling_items = FallingItems()
        self.new_level = False

        self.move_index = None
        self.move_end = None

        self.fps_calculator = FrameRateCalculator()
        self.map = Map(stay_timeout)
        self.stay_timeout = stay_timeout

    def init_window(self, outer_window_box=None):
        self.game_window.init_window(outer_window_box)
        self.window_size = self.game_window.window_size

        image = np.zeros((MAZE_HEIGHT, MAZE_WIDTH, 3), dtype=np.uint8)
        cv2.imshow("Monitor", image)
        cv2.moveWindow("Monitor", 0, 0)
        cv2.setWindowProperty("Monitor", cv2.WND_PROP_TOPMOST, 1)

        x1, y1, x2, y2 = self.game_window.window_box
        click(x2 - 10, y2 - 10)

    def reset(self):
        self.action_manager.reset()
        self.enemy_list = EnemyList()
        self.falling_items = FallingItems()

        self.move_index = None
        self.move_end = None

        self.update()
        self.render()

        map_index = self.game_window.get_map_index_at_start()
        self.map.reset(map_index)

        status = self.try_next_level()
        state = self.state_encode()
        print("游戏开始！大地图：", map_index)

        return state, status

    def update(self):
        status_code = self.game_window.update()
        start_time = time.time()
        while status_code == -1:
            time.sleep(0.1)
            status_code = self.game_window.update()

            if time.time() - start_time > self.timeout:
                raise TimeoutError("Loading timeout")

        maze = self.map.get_maze()

        player = Player(self.game_window.player_box, maze)
        if self.player is not None:
            if self.player.is_next(player):
                self.map.next_level()
                self.new_level = True
                print(f"进入第 {self.map.level} 关：关卡类型 {self.map.get_task()}")
        self.player = player

        self.bullets.update(self.game_window.bullet_list_normal, self.player)

        self.enemy_list.update(self.game_window.enemy_boxes, maze)
        self.enemy_list.sort_by(self.player)

        self.action_manager.update_pao(self.game_window.items["pao"])

        self.falling_items.update(self.game_window.falling_items_normal)
        self.falling_items.sort_by(self.player)

        current_time = time.time()
        cool_down = 2
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
                fired_enemy.set_fired(current_time, cool_down)

        if self.map.level > 0:
            self.map.update(self.player, self.bullets, self.enemy_list)

        return status_code

    def auto_step(self):
        self.try_fire()
        status_code = self.move_to_enemy_and_update()

        if status_code:
            return status_code

        self.try_start_level()
        status_code = self.try_next_level()

        return status_code

    def step(self, action=None):
        assert action is None

        self.try_fire()
        self.render()
        status_code = self.update()
        state = self.state_encode()

        return state, status_code

    def try_fire(self):
        if len(self.enemy_list) == 0:
            return -0.1

        direction = self.map.fire_direction(self.player, self.enemy_list)
        if direction is not None:
            fired = self.action_manager.fire_to(direction)
            return 0.1 + fired * 0.4
        return -0.1

    def try_start_level(self):
        def loop_act(direction):
            self.try_fire()
            self.action_manager.take_action_direction(direction)
            self.render()

            time.sleep(0.1)
            self.update()

        while self.new_level:
            if self.map.get_task() == 5:
                self.action_manager.take_action(0)
                time.sleep(1)
            elif self.map.map_index == 1:
                if self.map.level == 3:
                    self.action_manager.take_action(0)
                    time.sleep(1)
            elif self.map.map_index == 3:
                if self.map.level == 1:
                    direction = Direction.Down
                    self.action_manager.take_action_direction(direction)
                    time.sleep(1)
            elif self.map.map_index == 4:
                if self.map.level == 1:
                    loop_act(Direction.DownLeft)
                    if self.player.xn > 0.75:
                        continue
                elif self.map.level == 2:
                    direction = Direction.Left
                    loop_act(direction)
                    if self.player.xn > 0.3:
                        continue
            elif self.map.map_index == 5:
                if self.map.level == 2:
                    direction = Direction.Left
                    loop_act(direction)
                    if self.player.xn > 0.3:
                        continue

            self.new_level = False

    def can_next_level(self):
        return len(self.enemy_list) == 0 and len(self.bullets) == 0

    def try_next_level(self):
        loop_start_time = time.time()
        while self.can_next_level() and not self.new_level:
            direction, path = self.map.get_next_direction(self.player, self.falling_items)
            self.render(path)
            self.action_manager.take_action_direction(direction)
            time.sleep(0.1)
            status_code = self.update()

            if status_code:
                return status_code
            if time.time() - loop_start_time > self.stay_timeout:
                return -3

        return 0

    def move_to_enemy_and_update(self):
        if self.can_next_level():
            return 0
        if self.new_level:
            return 0

        start_time = time.time()
        direction, path, self.move_index = self.map.get_shot_place_direction(
            self.player, self.move_index, self.move_end
        )
        if direction is None:
            self.action_manager.take_action(0)
            self.render()
            status_code = self.update()
            return status_code

        self.move_end = path[-1]
        self.action_manager.take_action_direction(direction)
        self.render(path)

        delay_time = self.act_interval - (time.time() - start_time)
        if delay_time > 0:
            time.sleep(delay_time)

        status_code = self.update()

        return status_code

    def state_encode(self):
        state = np.zeros(self.state_space, dtype=np.float32)
        state[0] = self.player.xn
        state[1] = self.player.yn
        state[2] = self.action_manager.get_fire_cool_down()
        state[3] = self.game_window.hit_point
        pao, dan, fan = self.game_window.items.values()
        state[4] = np.array(pao, dtype=np.float32)
        state[5] = np.array(dan, dtype=np.float32)
        state[6] = np.array(fan, dtype=np.float32)

        state[7] = self.map.level
        task = self.map.get_task()
        state[8:17] = np.eye(9)[task]

        # 统一编码最近的六个敌人，保持状态向量总长度不变。
        for i, enemy in enumerate(self.enemy_list):
            if i >= 6:
                break
            state[17 + i * 3:20 + i * 3] = enemy.state_encode()
        state[35:63] = self.bullets.state_encode()

        return state

    def render(self, path=None):
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

        fps = self.fps_calculator.trigger()

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
        cv2.putText(image, f"FPS: {fps}", (250, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
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

        state, status = env.reset()
        while status == 0:
            status = env.auto_step()
            if status == 1:
                success_count += 1
        time.sleep(1)
        env.game_window.recover(status)


if __name__ == "__main__":
    from pathlib import Path
    import sys
    import os

    PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
    os.chdir(PROJECT_DIR)

    if len(sys.argv) > 1:
        target_ = int(sys.argv[1])
    else:
        target_ = 3

    env_ = Env()
    env_.init_window()
    auto_loop(env_, target_)
