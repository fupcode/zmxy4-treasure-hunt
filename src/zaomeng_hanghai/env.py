import numpy as np
import time
import cv2
from scipy.ndimage import binary_dilation
from enum import Enum
import win32api
import win32con
from collections import UserList

from zaomeng_hanghai.game_window import GameWindow
from zaomeng_hanghai.core import maze_utils

MAZE_HEIGHT = 200
MAZE_WIDTH = 320

HIT_HEIGHT = 12
HIT_WIDTH = 15

EXIT_LEFT = 0.46
EXIT_RIGHT = 0.57
EXIT_UP = 0.40
EXIT_DOWN = 0.54

MAP_LEFT = 0.066
MAP_RIGHT = 0.95
MAP_UP = 0.105
MAP_DOWN = 0.866

COLLISION_LEFT = 0.018
COLLISION_RIGHT = 0.012
COLLISION_UP = 0
COLLISION_DOWN = 0.045

EDGE_LEFT = MAP_LEFT + COLLISION_LEFT
EDGE_RIGHT = MAP_RIGHT - COLLISION_RIGHT
EDGE_UP = MAP_UP
EDGE_DOWN = MAP_DOWN - COLLISION_DOWN

END_LEFT = 0.463
END_RIGHT = 0.555
END_UP = 0.653
END_DOWN = 0.736

FIRE_X_RANGE = int(0.3 * MAZE_WIDTH)
FIRE_Y_RANGE = int(0.5 * MAZE_HEIGHT)


class Direction(Enum):
    Right = "D"
    Down = "S"
    Left = "A"
    Up = "W"
    DownRight = "SD"
    DownLeft = "SA"
    UpRight = "WD"
    UpLeft = "WA"


direction_list = [Direction.Right, Direction.Down, Direction.Left, Direction.Up,
                  Direction.DownRight, Direction.DownLeft, Direction.UpRight, Direction.UpLeft]
move_list = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]


# 耗时记录装饰器
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapse_time = time.time() - start_time
        if elapse_time > 0.02:
            print(f"Function '{func.__name__}' of Env executed in {elapse_time:.4f} seconds.")
        return result

    return wrapper


def key_down(string):
    """模拟按下键"""
    for char in string:
        vk_code = ord(char.upper())
        win32api.keybd_event(vk_code, 0, 0, 0)


def key_up(string):
    """模拟释放键"""
    for char in string:
        vk_code = ord(char.upper())
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)


def move_to(direction):
    if direction is None:
        return
    key_down(direction.value)


def stop_to(direction):
    if direction is None:
        return
    key_up(direction.value)


def key_up_all():
    for char in "WASD":
        key_up(char)


def click(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


class Player:
    def __init__(self, player_box, maze):
        xywhn = player_box.xywhn
        self.xn = xywhn[0]
        self.yn = xywhn[1] + xywhn[3] / 5

        x = int(self.xn * MAZE_WIDTH)
        y = int(self.yn * MAZE_HEIGHT)

        self.x, self.y = maze_utils.find_nearest_passable_point(maze, (x, y))

    def is_next(self, player):
        # 如果玩家坐标突变，说明玩家进入下一关
        if abs(self.xn - player.xn) > 0.5 or abs(self.yn - player.yn) > 0.5:
            return True
        return False

    def relative_center_position(self):
        dx = 0
        dy = 0
        if self.xn > EXIT_RIGHT:
            dx = 1
        elif self.xn < EXIT_LEFT:
            dx = -1
        if self.yn > EXIT_DOWN:
            dy = 1
        elif self.yn < EXIT_UP:
            dy = -1
        return dx, dy

    def in_forbidden_area(self):
        if 0.68 < self.xn < 0.87 and 0.07 < self.yn < 0.19:
            return True
        elif 0.86 < self.xn < 0.96 and 0.1 < self.yn < 0.25:
            return True
        elif 0.05 < self.xn < 0.1 and 0.07 < self.yn < 0.20:
            return True
        elif 0.07 < self.xn < 0.21 and 0.04 < self.yn < 0.16:
            return True
        return False


class Enemy:
    def __init__(self, enemy_box, maze):
        xywhn = enemy_box.xywhn
        self.xn = xywhn[0]
        self.yn = xywhn[1] + xywhn[3] / 6

        x = int(self.xn * MAZE_WIDTH)
        y = int(self.yn * MAZE_HEIGHT)

        self.x, self.y = maze_utils.find_nearest_passable_point(maze, (x, y))

        self.x1n, self.y1n, self.x2n, self.y2n = enemy_box.xyxyn
        self.y1n += xywhn[3] / 2

        self.x1 = int(self.x1n * MAZE_WIDTH)
        self.y1 = int(self.y1n * MAZE_HEIGHT)
        self.x2 = int(self.x2n * MAZE_WIDTH)
        self.y2 = int(self.y2n * MAZE_HEIGHT)

        self.disappear_count = 0
        self.fired = False
        self.fired_time = None
        self.cool_down_time = None

        self.state_dim = 3

        self.distance_array = self.get_distance_array()
        self.vision = self.calculate_vision(maze, FIRE_Y_RANGE, FIRE_X_RANGE)

    def inherit(self, enemy):
        self.fired = enemy.fired
        self.fired_time = enemy.fired_time
        self.cool_down_time = enemy.cool_down_time

    def get_distance_array(self):
        distance_array = np.abs(self.x - np.arange(MAZE_WIDTH)) + np.abs(
            self.y - np.arange(MAZE_HEIGHT)[:, np.newaxis])

        distance_array[self.y1:self.y2, :] = distance_array[self.y1:self.y2, :]
        distance_array[:, self.x1:self.x2] = distance_array[:, self.x1:self.x2]

        return distance_array

    def calculate_vision(self, map_array, vertical_length, horizontal_length, group_size=5):
        # 初始化视野数组，值默认为1
        vision_array = np.ones_like(map_array, dtype=np.float32)

        # 地图大小
        map_height, map_width = map_array.shape

        # 敌人 box 区域初始化为 0
        vision_array[self.y1:self.y2, self.x1:self.x2] = 0

        # 计算衰减值的函数
        def calculate_decay(distance, max_distance):
            return distance / max_distance

        # 向上扩展视野
        for x_start in range(self.x1, self.x2, group_size):
            x_end = min(x_start + group_size, self.x2)
            for dy in range(1, vertical_length + 1):
                y = self.y1 - dy
                if y < 0 or np.any(map_array[y, x_start:x_end] == 1):
                    break
                decay_value = calculate_decay(dy, vertical_length)
                vision_array[y, x_start:x_end] = decay_value

        # 向下扩展视野
        for x_start in range(self.x1, self.x2, group_size):
            x_end = min(x_start + group_size, self.x2)
            for dy in range(1, vertical_length + 1):
                y = self.y2 + dy - 1
                if y >= map_height or np.any(map_array[y, x_start:x_end] == 1):
                    break
                decay_value = calculate_decay(dy, vertical_length)
                vision_array[y, x_start:x_end] = decay_value

        # 向左扩展视野
        for y_start in range(self.y1, self.y2, group_size):
            y_end = min(y_start + group_size, self.y2)
            for dx in range(1, horizontal_length + 1):
                x = self.x1 - dx
                if x < 0 or np.any(map_array[y_start:y_end, x] == 1):
                    break
                decay_value = calculate_decay(dx, horizontal_length)
                vision_array[y_start:y_end, x] = decay_value

        # 向右扩展视野
        for y_start in range(self.y1, self.y2, group_size):
            y_end = min(y_start + group_size, self.y2)
            for dx in range(1, horizontal_length + 1):
                x = self.x2 + dx - 1
                if x >= map_width or np.any(map_array[y_start:y_end, x] == 1):
                    break
                decay_value = calculate_decay(dx, horizontal_length)
                vision_array[y_start:y_end, x] = decay_value

        return vision_array

    def set_fired(self, current_time, cool_down_time):
        self.fired = True
        self.fired_time = current_time
        self.cool_down_time = cool_down_time

    def is_safe(self, current_time):
        if self.fired:
            return current_time - self.fired_time < self.cool_down_time
        return False

    def state_encode(self):
        state = np.array([1, self.xn, self.yn])
        return state


class EnemyList(UserList):
    def __init__(self, max_disappear_count=1, threshold_distance=10):
        super().__init__()
        self.data = []
        self.max_disappear_count = max_disappear_count
        self.threshold_distance = threshold_distance

    def update(self, enemy_box_list, maze):
        new_enemies = [Enemy(enemy_box, maze) for enemy_box in enemy_box_list]
        matched = [False] * len(self.data)

        for new_enemy in new_enemies:
            found_match = False
            for i, existing_enemy in enumerate(self.data):
                dist = abs(new_enemy.x - existing_enemy.x) + abs(new_enemy.y - existing_enemy.y)
                if dist <= self.threshold_distance:
                    found_match = True
                    matched[i] = True
                    new_enemy.inherit(existing_enemy)
                    self.data[i] = new_enemy
                    break

            if not found_match:
                self.data.append(new_enemy)
                matched.append(True)

        for i, enemy in enumerate(self.data):
            if not matched[i]:
                enemy.disappear_count += 1

        self.data = [enemy for enemy in self.data if enemy.disappear_count <= self.max_disappear_count]

    def sort_by(self, player):
        self.data.sort(key=lambda enemy: abs(enemy.x - player.x) + abs(enemy.y - player.y))

    def __add__(self, other):
        # 确保 `other` 是一个可迭代对象，比如列表或另一个 UserList
        if isinstance(other, (list, UserList)):
            # 返回一个新的 CustomList 对象，包含合并后的元素
            return self.data + list(other)
        raise TypeError(f"Unsupported operand type(s) for +: 'CustomList' and '{type(other).__name__}'")

    def __radd__(self, other):
        # 实现反向加法，确保列表在左侧时也可以加上 CustomList
        if isinstance(other, list):
            return list(other) + self.data
        raise TypeError(f"Unsupported operand type(s) for +: '{type(other).__name__}' and 'CustomList'")


class ActionManager:
    def __init__(self):
        self.fire_interval = 0.6
        self.pao = False
        self.last_fire_time = time.time()
        self.last_direction = None

    def reset(self):
        self.fire_interval = 0.6
        self.pao = False
        self.last_fire_time = time.time()

    def _can_fire(self):
        if time.time() - self.last_fire_time > self.fire_interval:
            self.last_fire_time = time.time()
            return True
        return False

    def get_fire_cool_down(self):
        return max(0.0, self.last_fire_time + self.fire_interval - time.time())

    def get_no_fire_time(self):
        return time.time() - self.last_fire_time + self.fire_interval

    def update_pao(self, pao):
        if self.pao == pao:
            return

        self.pao = pao
        if pao:
            self.fire_interval = 0.4
        else:
            self.fire_interval = 0.6

    def double_moving(self):
        return len(self.last_direction.value) == 2

    def single_conflict_direction(self, single_direction):
        if len(single_direction.value) != 1:
            raise ValueError("Invalid direction value.")
        if self.last_direction is None:
            return None, single_direction

        if self.last_direction == Direction.DownRight:
            if single_direction == Direction.Right:
                return None, None
            return Direction.Right, single_direction
        elif self.last_direction == Direction.DownLeft:
            if single_direction == Direction.Left:
                return None, None
            return Direction.Left, single_direction
        elif self.last_direction == Direction.UpRight:
            if single_direction == Direction.Right:
                return None, None
            return Direction.Right, single_direction
        elif self.last_direction == Direction.UpLeft:
            if single_direction == Direction.Left:
                return None, None
            return Direction.Left, single_direction
        elif self.last_direction == Direction.Right or self.last_direction == Direction.Left:
            if single_direction == Direction.Down or single_direction == Direction.Up:
                return self.last_direction, single_direction
            elif self.last_direction == single_direction:
                return None, None
            else:
                return None, single_direction
        elif self.last_direction == Direction.Down or self.last_direction == Direction.Up:
            return None, single_direction

    def fire_to(self, direction):
        if self._can_fire():
            suspend_direction, move_direction = self.single_conflict_direction(direction)
            stop_to(suspend_direction)
            move_to(move_direction)
            time.sleep(0.005)
            move_to(move_direction)
            time.sleep(0.005)
            move_to(move_direction)
            time.sleep(0.005)
            key_down("J")
            key_up("J")
            stop_to(move_direction)
            move_to(suspend_direction)
            return 1
        return 0

    def modify_direction(self, direction):
        if self.last_direction is None:
            return None, direction
        if direction == self.last_direction:  # 完全符合
            return None, None
        # 处理部分冲突
        if self.double_moving():
            if len(direction.value) == 2:  # 双双部分冲突
                if self.last_direction == Direction.DownRight:
                    if direction == Direction.DownLeft:
                        return Direction.Right, Direction.Left
                    elif direction == Direction.UpRight:
                        return Direction.Down, Direction.Up
                elif self.last_direction == Direction.DownLeft:
                    if direction == Direction.DownRight:
                        return Direction.Left, Direction.Right
                    elif direction == Direction.UpLeft:
                        return Direction.Down, Direction.Up
                elif self.last_direction == Direction.UpRight:
                    if direction == Direction.UpLeft:
                        return Direction.Right, Direction.Left
                    elif direction == Direction.DownRight:
                        return Direction.Up, Direction.Down
                elif self.last_direction == Direction.UpLeft:
                    if direction == Direction.UpRight:
                        return Direction.Left, Direction.Right
                    elif direction == Direction.DownLeft:
                        return Direction.Up, Direction.Down
            else:  # 双单部分冲突
                if self.last_direction == Direction.DownRight:
                    if direction == Direction.Right:
                        return Direction.Down, None
                    elif direction == Direction.Down:
                        return Direction.Right, None
                elif self.last_direction == Direction.DownLeft:
                    if direction == Direction.Left:
                        return Direction.Down, None
                    elif direction == Direction.Down:
                        return Direction.Left, None
                elif self.last_direction == Direction.UpRight:
                    if direction == Direction.Right:
                        return Direction.Up, None
                    elif direction == Direction.Up:
                        return Direction.Right, None
                elif self.last_direction == Direction.UpLeft:
                    if direction == Direction.Left:
                        return Direction.Up, None
                    elif direction == Direction.Up:
                        return Direction.Left, None
        else:  # 单双部分冲突，没有单单部分冲突
            if len(direction.value) == 2:
                if self.last_direction == Direction.Right:
                    if direction == Direction.DownRight:
                        return None, Direction.Down
                    elif direction == Direction.UpRight:
                        return None, Direction.Up
                elif self.last_direction == Direction.Left:
                    if direction == Direction.DownLeft:
                        return None, Direction.Down
                    elif direction == Direction.UpLeft:
                        return None, Direction.Up
                elif self.last_direction == Direction.Down:
                    if direction == Direction.DownRight:
                        return None, Direction.Right
                    elif direction == Direction.DownLeft:
                        return None, Direction.Left
                elif self.last_direction == Direction.Up:
                    if direction == Direction.UpRight:
                        return None, Direction.Right
                    elif direction == Direction.UpLeft:
                        return None, Direction.Left
        # 完全冲突
        return self.last_direction, direction

    def take_action_direction(self, direction):
        stop_direction, move_direction = self.modify_direction(direction)
        stop_to(stop_direction)
        move_to(move_direction)
        move_to(direction)
        self.last_direction = direction

    def take_action(self, action):
        if action == 0:
            stop_to(self.last_direction)
            self.last_direction = None
            return
        direction = direction_list[action - 1]
        self.take_action_direction(direction)


class Bullet:
    def __init__(self, bullet_info):
        self.state_dim = 7
        self.xn, self.yn, self.d = bullet_info
        self.x = int(self.xn * MAZE_WIDTH)
        self.y = int(self.yn * MAZE_HEIGHT)

        self.disappear_count = 0

    def state_encode(self):
        state = np.zeros(self.state_dim)
        state[0] = 1
        state[1 + self.d] = 1
        state[5] = self.xn
        state[6] = self.yn
        return state


class Bullets(UserList):
    def __init__(self, disappear_threshold=2):
        super().__init__()
        self.state_dim = 7 * 4
        self.data = []
        self.safe_bullets = []
        self.new_bullets = []

        self.disappear_threshold = disappear_threshold

    def update(self, bullet_list_normal, player):
        bullets = []
        safe_bullets = []
        new_bullets = []
        for bullet_info in bullet_list_normal:
            bullet = Bullet(bullet_info)
            if self.is_safe(bullet, player):
                safe_bullets.append(bullet)
            else:
                bullets.append(bullet)

        # 按照子弹离玩家距离排序
        bullets.sort(key=lambda bullet: abs(bullet.x - player.x) + abs(bullet.y - player.y))

        matched = [False] * len(self.data)
        for bullet in bullets:
            for i, old_bullet in enumerate(self.data):
                if bullet.d != old_bullet.d:
                    continue
                if abs(bullet.y - old_bullet.y) <= 5:
                    if bullet.d == 0 and 0 <= bullet.x - old_bullet.x <= FIRE_X_RANGE:
                        matched[i] = True
                        self.data[i] = bullet
                        break
                    elif bullet.d == 2 and 0 <= old_bullet.x - bullet.x <= FIRE_X_RANGE:
                        matched[i] = True
                        self.data[i] = bullet
                        break
                elif abs(bullet.x - old_bullet.x) <= 5:
                    if bullet.d == 1 and 0 <= bullet.y - old_bullet.y <= FIRE_Y_RANGE:
                        matched[i] = True
                        self.data[i] = bullet
                        break
                    elif bullet.d == 3 and 0 <= old_bullet.y - bullet.y <= FIRE_Y_RANGE:
                        matched[i] = True
                        self.data[i] = bullet
                        break
            else:
                self.data.append(bullet)
                matched.append(True)
                new_bullets.append(bullet)

        for i, bullet in enumerate(self.data):
            if not matched[i]:
                bullet.disappear_count += 1

        self.data = [bullet for bullet in self.data if bullet.disappear_count <= self.disappear_threshold]
        self.safe_bullets = safe_bullets
        self.new_bullets = new_bullets

    def is_safe(self, bullet, player):
        if bullet.d == 0 and bullet.x > player.x + 6:
            return True
        elif bullet.d == 1 and bullet.y > player.y + 4:
            return True
        elif bullet.d == 2 and bullet.x < player.x - 6:
            return True
        elif bullet.d == 3 and bullet.y < player.y - 4:
            return True
        return False

    def state_encode(self):
        state = np.zeros(self.state_dim)
        for i, bullet in enumerate(self.data):
            if i >= 4:
                break
            state[i * 7:(i + 1) * 7] = bullet.state_encode()
        return state


class FallingItem:
    def __init__(self, item_pos_normal):
        self.xn, self.yn = item_pos_normal

        self.x = int(self.xn * MAZE_WIDTH)
        self.y = int(self.yn * MAZE_HEIGHT)

        self.disappear_count = 0


class FallingItems(UserList):
    def __init__(self, max_disappear_count=3, threshold_distance=10):
        super().__init__()
        self.data = []
        self.max_disappear_count = max_disappear_count
        self.threshold_distance = threshold_distance

    def update(self, item_list_normal):
        new_items = [FallingItem(pos) for pos in item_list_normal]
        matched = [False] * len(self.data)  # 标记已有物品是否被匹配到新物品

        # 遍历新物品
        for new_item in new_items:
            found_match = False
            for i, existing_item in enumerate(self.data):
                # 计算新物品与已有物品的曼哈顿距离
                dist = abs(new_item.x - existing_item.x) + abs(new_item.y - existing_item.y)
                if dist <= self.threshold_distance:
                    # 如果距离小于阈值，认为是同一个物品
                    found_match = True
                    matched[i] = True  # 标记为匹配到
                    self.data[i] = new_item  # 更新物品
                    break

            if not found_match:
                # 如果没有找到匹配的老物品，将其视为新物品加入
                self.data.append(new_item)
                matched.append(True)  # 为新加入的物品补充匹配标记

        # 检查未匹配到的老物品并增加 disappear_count
        for i, item in enumerate(self.data):
            if not matched[i]:
                item.disappear_count += 1

        # 删除超过最大消失计数的物品
        self.data = [item for item in self.data if item.disappear_count <= self.max_disappear_count]

    def sort_by(self, player):
        # 根据与玩家的曼哈顿距离排序
        self.data.sort(key=lambda item: abs(item.x - player.x) + abs(item.y - player.y))


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


class Map:
    def __init__(self, stay_time_out):
        # 三个常量
        self.route_list = [
            (1, 1, 2, 3, 4),
            (0, 0, 3, 3, 4),
            (2, 3, 0, 0, 4),
            (3, 0, 1, 1, 4),
            (2, 3, 3, 0, 4),
            (3, 3, 2, 2, 4),
            (1, 0, 0, 1, 4)
        ]

        self.tasks_list = [
            (0, 1, 2, 3),
            (4, 5, 0, 3),
            (1, 6, 7, 3),
            (2, 8, 4, 3),
            (6, 8, 1, 3),
            (5, 8, 7, 3),
            (6, 7, 2, 3),
        ]
        self.task_maps = np.zeros((9, 7, 13), dtype=np.bool_)

        # 两个个变量，reset时初始化为当前map_index对应的值
        self.route = None

        self.map_index = None
        self.level = 0
        self.next_exit = None

        self.collision_left = None
        self.collision_right = None
        self.collision_up = None
        self.collision_down = None

        self.map_left = None
        self.map_up = None
        self.map_width = None
        self.map_height = None

        self.map_template = None

        self.init_task_maps()
        self.map_template, self.map_template_without_exit = self.init_map_template((MAZE_WIDTH, MAZE_HEIGHT))

        self.level_mazes = None
        self.distance_mazes = None
        self.stay_time_out = stay_time_out
        self.enter_time = None

        self.debug_image = None

    def init_task_maps(self):
        self.task_maps[0] = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=np.bool_)

        self.task_maps[1][0][-1] = 1
        self.task_maps[1][2][3] = 1
        self.task_maps[1][3][5] = 1
        self.task_maps[1][3][7] = 1
        self.task_maps[1][4][9] = 1
        self.task_maps[1][6][0] = 1

        self.task_maps[2] = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.bool_)

        self.task_maps[3][1][-2] = 1
        self.task_maps[3][-2][1] = 1

        self.task_maps[4] = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.bool_)

        self.task_maps[5] = np.array([
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        ], dtype=np.bool_)

        self.task_maps[6] = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ], dtype=np.bool_)

        self.task_maps[7] = np.array([
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        ], dtype=np.bool_)

        self.task_maps[8] = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.bool_)

    def init_map_template(self, maze_size):
        width, height = maze_size
        map_template = np.zeros((height, width), dtype=np.bool_)
        exit_left = int(width * EXIT_LEFT)
        exit_right = int(width * EXIT_RIGHT)
        exit_up = int(height * EXIT_UP)
        exit_down = int(height * EXIT_DOWN)

        edge_left = int(width * EDGE_LEFT)
        edge_right = int(width * EDGE_RIGHT)
        edge_up = int(height * EDGE_UP)
        edge_down = int(height * EDGE_DOWN)

        map_template[:edge_up, :] = 1
        map_template[edge_down:, :] = 1
        map_template[:, :edge_left] = 1
        map_template[:, edge_right:] = 1

        map_template_without_exit = map_template.copy()

        map_template[exit_up:exit_down, :] = 0
        map_template[:, exit_left:exit_right] = 0

        left_top = (int(0.22 * MAZE_WIDTH), int(0.18 * MAZE_HEIGHT))
        right_top1 = (int(0.68 * MAZE_WIDTH), int(0.19 * MAZE_HEIGHT))
        right_top2 = (int(0.86 * MAZE_WIDTH), int(0.25 * MAZE_HEIGHT))

        map_template_without_exit[:left_top[1], :left_top[0]] = 1
        map_template_without_exit[:right_top1[1], right_top1[0]:] = 1
        map_template_without_exit[:right_top2[1], right_top2[0]:] = 1

        return map_template, map_template_without_exit

    def reset(self, map_index):
        self.map_index = map_index
        self.route = self.route_list[map_index]

        self.level = 0
        self.next_exit = self.route[self.level]
        self.level_mazes, self.distance_mazes = self.reset_map_list((MAZE_WIDTH, MAZE_HEIGHT), self.map_template)

        self.enter_time = time.time()

    def reset_map_list(self, maze_size, map_template):
        level_mazes = []
        distance_mazes = []
        width, height = maze_size
        tasks = self.tasks_list[self.map_index]

        map_left = int(MAP_LEFT * width)
        map_up = int(MAP_UP * height)
        map_width = int((MAP_RIGHT - MAP_LEFT) * width)
        map_height = int((MAP_DOWN - MAP_UP) * height)

        collision_left = int(maze_size[0] * COLLISION_LEFT)
        collision_right = int(maze_size[0] * COLLISION_RIGHT)
        collision_up = int(maze_size[1] * COLLISION_UP)
        collision_down = int(maze_size[1] * COLLISION_DOWN)

        for task in tasks:
            maze = np.zeros((height, width), dtype=np.bool_)
            map_area = self.enlarge_bool_array(self.task_maps[task], map_width, map_height)

            maze[map_up:map_up + map_height, map_left:map_left + map_width] = map_area
            maze = self.directional_dilate(maze, collision_right, collision_left, collision_down, collision_up)

            maze = np.logical_or(maze, map_template)
            distance_maze = maze_utils.calculate_distance_from_obstacles(maze)

            level_mazes.append(maze)
            distance_mazes.append(distance_maze)

        return level_mazes, distance_mazes

    def get_maze(self):
        if self.level == 0:
            return self.map_template
        return self.level_mazes[self.level - 1]

    def get_task(self):
        if self.level == 0:
            return -1
        return self.tasks_list[self.map_index][self.level - 1]

    def is_stay_time_out(self):
        if time.time() - self.enter_time > self.stay_time_out:
            return True
        return False

    def get_distance_maze(self):
        if self.level == 0:
            return self.map_template_without_exit
        return self.distance_mazes[self.level - 1]

    def enlarge_bool_array(self, task_map, map_width, map_height):
        original_height, original_width = task_map.shape

        # 计算每个元素扩展后的矩阵大小
        factor_x = map_width // original_width + 1
        factor_y = map_height // original_height + 1

        # 扩展每个元素
        enlarged_map = np.block([[np.full((factor_y, factor_x), val) for val in row] for row in task_map])

        # 计算需要裁剪的上下左右的数量
        extra_height = enlarged_map.shape[0] - map_height
        extra_width = enlarged_map.shape[1] - map_width

        # 均匀裁剪上下左右
        crop_top = extra_height // 2
        crop_bottom = extra_height - crop_top
        crop_left = extra_width // 2
        crop_right = extra_width - crop_left

        cropped_map = enlarged_map[crop_top:-crop_bottom, crop_left:-crop_right]

        return cropped_map

    def directional_dilate(self, bool_array, expand_left=0, expand_right=0, expand_top=0, expand_bottom=0):
        # 根据指定的方向构建结构元素（卷积核）
        half_height = max(expand_top, expand_bottom)
        half_width = max(expand_left, expand_right)
        kernel_height = 2 * half_height + 1
        kernel_width = 2 * half_width + 1
        kernel = np.zeros((kernel_height, kernel_width), dtype=np.bool_)

        # 将指定方向的区域填充为 True
        kernel[half_height - expand_top:half_height + expand_bottom + 1,
        half_width - expand_left:half_width + expand_right + 1] = True

        # 使用 binary_dilation 进行膨胀
        expanded_array = binary_dilation(bool_array, structure=kernel, output=True)

        return expanded_array

    def next_level(self):
        self.level += 1
        self.enter_time = time.time()
        if self.level == 5:
            self.level = 0

        self.next_exit = self.route[self.level]

    def fire_distance(self, player, enemy):
        maze = self.get_maze()
        if enemy.x2 > player.x > enemy.x1:
            if abs(player.yn - enemy.yn) > 0.52:
                return None, None
            if self.is_collision_col(maze, enemy.y, player.y, player.x):
                return enemy.y - player.y, 0
        elif enemy.y2 > player.y > enemy.y1:
            if abs(player.xn - enemy.xn) > 0.33:
                return None, None
            if self.is_collision_row(maze, enemy.x, player.x, player.y):
                return enemy.x - player.x, 1
        return None, None

    def fire_direction(self, player, enemy_list):
        min_distance = MAZE_HEIGHT + MAZE_WIDTH
        direction = None

        for enemy in enemy_list:
            if enemy is None:
                continue
            distance, row = self.fire_distance(player, enemy)
            if distance is not None:
                if abs(distance) < min_distance:
                    min_distance = abs(distance)
                    if row:
                        if distance > 0:
                            direction = Direction.Right
                        else:
                            direction = Direction.Left
                    else:
                        if distance > 0:
                            direction = Direction.Down
                        else:
                            direction = Direction.Up

        return direction

    def is_collision_row(self, maze, x1, x2, y):
        if x1 > x2:
            return not np.any(maze[y, x2:x1])
        else:
            return not np.any(maze[y, x1:x2])

    def is_collision_col(self, maze, y1, y2, x):
        if y1 > y2:
            return not np.any(maze[y2:y1, x])
        else:
            return not np.any(maze[y1:y2, x])

    def collision_row_range(self, maze, x, left, right, y):
        row = maze[y]
        # 向左搜索
        left_pos = -1
        for i in range(x, left, -1):
            if row[i] != 0:
                left_pos = i
                break

        # 向右搜索
        right_pos = -1
        for i in range(x, right):
            if row[i] != 0:
                right_pos = i
                break

        # 如果没有找到非零元素，则返回原范围
        if left_pos == -1 and right_pos == -1:
            return left, right

        # 返回找到的结果
        return left_pos if left_pos != -1 else left, right_pos if right_pos != -1 else right

    def collision_col_range(self, maze, y, up, down, x):
        col = maze[:, x]
        # 向上搜索
        up_pos = -1
        for i in range(y, up, -1):
            if col[i] != 0:
                up_pos = i
                break

        # 向下搜索
        down_pos = -1
        for i in range(y, down):
            if col[i] != 0:
                down_pos = i
                break

        # 如果没有找到非零元素，则返回原范围
        if up_pos == -1 and down_pos == -1:
            return up, down

        # 返回找到的结果
        return up_pos if up_pos != -1 else up, down_pos if down_pos != -1 else down

    def is_move_collision(self, player, action):
        dx, dy = move_list[action - 1]
        dx *= int(0.012 * MAZE_WIDTH)
        dy *= int(0.016 * MAZE_HEIGHT)
        target_point = (player.x + dx, player.y + dy)
        if self.get_maze()[target_point[1], target_point[0]] or self.map_template_without_exit[
            target_point[1], target_point[0]]:
            return True
        return False

    def get_direction_from_next_point(self, player, next_point):
        if next_point[0] > player.x:
            if next_point[1] > player.y:
                return Direction.DownRight
            elif next_point[1] < player.y:
                return Direction.UpRight
            else:
                return Direction.Right
        elif next_point[0] < player.x:
            if next_point[1] > player.y:
                return Direction.DownLeft
            elif next_point[1] < player.y:
                return Direction.UpLeft
            else:
                return Direction.Left
        else:
            if next_point[1] > player.y:
                return Direction.Down
            else:
                return Direction.Up

    def get_next_direction(self, player, falling_items):
        direction = self.nearly_exit(player)
        if direction is not None:
            return direction, []

        if self.level == 4:
            end = (MAZE_WIDTH * (END_LEFT + END_RIGHT) / 2, MAZE_HEIGHT * (END_UP + END_DOWN) / 2)
        else:
            if falling_items:
                end = (falling_items[0].x, falling_items[0].y)
            elif self.next_exit == 0:
                end = (MAZE_WIDTH - 1, MAZE_HEIGHT * (EXIT_UP + EXIT_DOWN) / 2)
            elif self.next_exit == 1:
                end = (MAZE_WIDTH * (EXIT_LEFT + EXIT_RIGHT) / 2, MAZE_HEIGHT - 1)
            elif self.next_exit == 2:
                end = (0, MAZE_HEIGHT * (EXIT_UP + EXIT_DOWN) / 2)
            else:
                end = (MAZE_WIDTH * (EXIT_LEFT + EXIT_RIGHT) / 2, 0)

        start = (player.x, player.y)
        path = maze_utils.find_path_to_point(self.get_maze(), start, end, distance_maze=self.get_distance_maze(),
                                             hate_factor=8)
        next_point = path[0]
        direction = self.get_direction_from_next_point(player, next_point)
        return direction, path

    def nearly_exit(self, player):
        dx, dy = player.relative_center_position()
        if self.level == 0:
            if self.next_exit in {1, 3}:
                if dx == 1:
                    return Direction.Left
                elif dx == -1:
                    return Direction.Right
                else:
                    return direction_list[self.next_exit]
            elif self.next_exit in {0, 2}:
                if dy == 1:
                    return Direction.Up
                elif dy == -1:
                    return Direction.Down
                else:
                    return direction_list[self.next_exit]

        elif self.level == 4:
            if END_LEFT < player.xn < END_RIGHT and END_UP < player.yn < END_DOWN:
                return Direction.Up
        return None

    def update(self, player, bullets, enemy_list):
        self.flee_mazes = self.get_flee_maze(player, bullets, enemy_list)

    def get_flee_maze(self, player, bullets, enemy_list):
        origin_maze = self.get_maze()
        moveable_maze = np.logical_or(origin_maze, self.map_template_without_exit)
        obstacle_distance_maze = self.get_distance_maze()
        end_region = np.ones((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.float32)

        hit_height = HIT_HEIGHT
        hit_width = HIT_WIDTH

        flee_region_bullet = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.int64)
        if self.level != 4:
            bullet_const = 0
        else:
            bullet_const = 50
        for bullet in bullets:
            range_left = max(0, bullet.x - FIRE_X_RANGE)
            range_right = min(MAZE_WIDTH, bullet.x + FIRE_X_RANGE)
            range_up = max(0, bullet.y - FIRE_Y_RANGE)
            range_down = min(MAZE_HEIGHT, bullet.y + FIRE_Y_RANGE)
            if bullet.d == 0:
                flee_region_bullet[bullet.y - hit_height:bullet.y + hit_height, bullet.x:range_right] = np.arange(
                    range_right - bullet.x, 0, -1) + bullet_const
            elif bullet.d == 1:
                flee_region_bullet[bullet.y:range_down, bullet.x - hit_width:bullet.x + hit_width] = \
                    np.arange(range_down - bullet.y, 0, -1)[..., np.newaxis] + bullet_const
            elif bullet.d == 2:
                flee_region_bullet[bullet.y - hit_height:bullet.y + hit_height, range_left:bullet.x] = np.arange(1,
                                                                                                                 bullet.x - range_left + 1) + bullet_const
            elif bullet.d == 3:
                flee_region_bullet[range_up:bullet.y, bullet.x - hit_width:bullet.x + hit_width] = \
                    np.arange(1, bullet.y - range_up + 1)[..., np.newaxis] + bullet_const
            moveable_maze[bullet.y - hit_height:bullet.y + hit_height, bullet.x - hit_width:bullet.x + hit_width] = 1
            obstacle_distance_maze[bullet.y - 4 * hit_height:bullet.y + 4 * hit_height,
                                   bullet.x - 4 * hit_width:bullet.x + 4 * hit_width] = 0
        kernel = np.ones((15, 15), np.uint8)
        larger_flee_region_bullet = binary_dilation(flee_region_bullet, kernel, iterations=1)

        flee_region_enemy = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.int64)
        if self.level != 4:
            flee_const = 45
        else:
            flee_const = 60
        for enemy in enemy_list:
            if enemy is not None:
                enemy_distance_r = flee_const - enemy.distance_array
                enemy_distance_r[enemy_distance_r < 0] = 0
                flee_region_enemy += enemy_distance_r

        peek_region_enemy = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.int64)
        peek_cross_enemy = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.int64)
        current_time = time.time()
        peek_x = 65
        peek_y = 50
        for enemy in enemy_list:
            if enemy is None or enemy.is_safe(current_time):
                continue
            range_left = max(0, enemy.x1 - peek_x)
            range_right = min(MAZE_WIDTH, enemy.x2 + peek_x)
            range_up = max(0, enemy.y1 - peek_y)
            range_down = min(MAZE_HEIGHT, enemy.y2 + peek_y)

            mask = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
            mask[range_up:range_down, range_left:range_right] = True
            peek_region_enemy[mask] += np.max(enemy.distance_array[mask]) - enemy.distance_array[mask]

            mask = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=bool)
            mask[enemy.y1:enemy.y2, range_left:range_right] = True
            mask[range_up:range_down, enemy.x1:enemy.x2] = True
            peek_cross_enemy[mask] += np.max(enemy.distance_array[mask]) - enemy.distance_array[mask]

        for enemy in enemy_list:
            if enemy is not None:
                if enemy.is_safe(current_time):
                    end_region[enemy.vision < 1] *= 0.92
                else:
                    end_region *= enemy.vision
                    if self.level == 4:
                        end_region[:enemy.y2, enemy.x1:enemy.x2] = 0

        end_region[end_region == 1] = 0
        distance_array = np.abs(player.x - np.arange(MAZE_WIDTH)) + np.abs(
            player.y - np.arange(MAZE_HEIGHT)[:, np.newaxis])
        decay_array = MAZE_HEIGHT / (distance_array + MAZE_HEIGHT)
        end_region *= decay_array
        end_region[larger_flee_region_bullet > 0] = 0
        end_region[flee_region_enemy > 0] = 0
        end_peek_region = np.zeros((MAZE_HEIGHT, MAZE_WIDTH), dtype=np.float32)
        mask = peek_cross_enemy > 0
        end_peek_region[mask] = end_region[mask]

        """
        flee_region_bullet区域为黄色
        flee_region_enemy区域为粉色
        end_peek_region区域为黄色
        end_region区域为紫色
        """
        self.debug_image = np.stack([255 * end_region, end_region, 255 * end_region], axis=-1).astype(np.uint8)
        mask = end_peek_region > 0
        self.debug_image[mask] = np.stack([end_peek_region, 255 * end_peek_region, 255 * end_region], axis=-1).astype(np.uint8)[mask]
        flee_region_enemy_image = flee_region_enemy / (np.max(flee_region_enemy) + 1)
        mask = flee_region_enemy > 0
        self.debug_image[mask] = np.stack([flee_region_enemy_image, flee_region_enemy_image, 255 * flee_region_enemy_image], axis=-1).astype(np.uint8)[mask]
        flee_region_bullet_image = flee_region_bullet / (np.max(flee_region_bullet) + 1)
        mask = flee_region_bullet > 0
        self.debug_image[mask] = np.stack([flee_region_bullet_image, 255 * flee_region_bullet_image, 255 * flee_region_bullet_image], axis=-1).astype(np.uint8)[mask]

        return moveable_maze, obstacle_distance_maze, flee_region_bullet, flee_region_enemy, peek_region_enemy, peek_cross_enemy, end_peek_region, end_region

    def get_shot_place_direction(self, player, last_move_index, last_move_end):
        moveable_maze, obstacle_distance_maze, flee_region_bullet, flee_region_enemy, peek_region_enemy, peek_cross_enemy, end_peek_region, end_region = self.flee_mazes
        flee_region = flee_region_bullet + flee_region_enemy

        # 逃离子弹和敌人
        if flee_region_bullet[player.y, player.x] > 0:
            if last_move_index != 1:
                last_move_end = None
            flee_path = maze_utils.find_nearest_passable_path(moveable_maze, (player.x, player.y), flee_region_bullet,
                                                              distance_maze=obstacle_distance_maze, hate_factor=5,
                                                              last_move_end=last_move_end)
            if len(flee_path) > 0:
                flee_direction = self.get_direction_from_next_point(player, flee_path[0])
                return flee_direction, flee_path, 1
            # else:
            #     print("未找到逃离路径")

        moveable_maze[flee_region > 0] = 1

        if self.level != 4:
            # peek后逃离
            if peek_cross_enemy[player.y, player.x] > 0:
                flee_path = maze_utils.find_nearest_passable_path(moveable_maze, (player.x, player.y), peek_cross_enemy,
                                                                  distance_maze=obstacle_distance_maze, hate_factor=8)
                if len(flee_path) > 0:
                    flee_direction = self.get_direction_from_next_point(player, flee_path[0])
                    return flee_direction, flee_path, 2
                # else:
                #     print("未找到peek逃离路径")

            # peek敌人
            if peek_region_enemy[player.y, player.x] > 0:
                peek_maze = moveable_maze.copy()
                peek_maze[peek_region_enemy == 0] = 1
                path = maze_utils.find_best_path_to_region(peek_maze, (player.x, player.y), end_peek_region,
                                                           distance_maze=obstacle_distance_maze, hate_factor=5,
                                                           stride=5)
                if len(path) > 0:
                    direction = self.get_direction_from_next_point(player, path[0])
                    return direction, path, 3
                # else:
                #     print("未找到peek路径")

            stride = 5
        else:
            stride = 10

        # 在远处时，去往敌人十字区域
        if end_region[player.y, player.x] > 0:
            return None, [], 0

        path = maze_utils.find_best_path_to_region(moveable_maze, (player.x, player.y), end_region,
                                                   distance_maze=obstacle_distance_maze, hate_factor=5, stride=stride)

        if len(path) > 0:
            next_point = path[0]
            direction = self.get_direction_from_next_point(player, next_point)
            return direction, path, 4
        # else:
        #     print("未找到远程射击路径")
        return None, [], -1


class Env:
    def __init__(self, time_out=3, stay_time_out=180, act_interval=0.02):
        self.timeout = time_out
        self.game_window = GameWindow()
        self.window_size = None
        self.action_manager = ActionManager()
        self.action_space = 9
        self.state_space = 7 + 10 + 3 * 6 + 7 * 4
        self.act_interval = act_interval

        self.player = None
        self.enemy_list = None
        self.boss = None
        self.bullets = Bullets()
        self.falling_items = None
        self.new_level = False

        self.fps_calculator = FrameRateCalculator()
        self.map = Map(stay_time_out)
        self.stay_time_out = stay_time_out

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

        self.update()

        map_index = self.game_window.reset()
        self.map.reset(map_index)

        status = self.try_next_level()
        state = self.state_encode()

        return state, status

    def update(self):
        status_code = self.wait_loading()

        maze = self.map.get_maze()

        player = Player(self.game_window.player_box, maze)
        if self.player is not None:
            if self.player.is_next(player):
                self.map.next_level()
                self.new_level = True
        self.player = player

        self.bullets.update(self.game_window.bullet_list_normal, self.player)

        self.enemy_list.update(self.game_window.enemy_boxes, maze)
        self.enemy_list.sort_by(self.player)
        self.boss = None
        if self.game_window.boss_box is not None:
            self.boss = Enemy(self.game_window.boss_box, maze)

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
            self.map.update(self.player, self.bullets, self.enemy_list + [self.boss])

        return status_code

    def wait_loading(self):
        start_time = time.time()
        status_code = self.game_window.update()
        while status_code == -1:
            time.sleep(0.1)
            status_code = self.game_window.update()

            if time.time() - start_time > self.timeout:
                raise TimeoutError("Loading timeout")

        return status_code

    def step(self, action):
        reward = 0

        hit_point = self.game_window.hit_point
        level = self.map.level
        self.action_manager.take_action(action)

        start_time = time.time()
        self.render()

        fire_reward = self.fire()
        reward += fire_reward
        if self.map.is_move_collision(self.player, action):
            reward -= 0.4

        self.move_to_enemy_and_update()
        delay_time = self.act_interval - (time.time() - start_time)
        if delay_time > 0:
            time.sleep(delay_time)
        status_code = self.update()

        if self.game_window.hit_point < hit_point:
            reward -= 0.8
        if self.player.in_forbidden_area():
            reward -= 0.4
        for enemy in self.enemy_list + [self.boss]:
            if enemy is not None:
                if enemy.x1 < self.player.x < enemy.x2 and enemy.y1 < self.player.y < enemy.y2:
                    reward -= 0.4

        if status_code:
            if status_code == 1:
                reward += 1
            elif status_code == -2:
                reward -= 1
            state = self.state_encode()
            return state, reward, status_code
        elif self.map.is_stay_time_out():
            status_code = -3  # 超时
            state = self.state_encode()
            reward -= 1
            return state, reward, status_code

        status_code = self.try_next_level()
        if status_code == 1:
            reward += 1
        elif status_code == -2:
            reward -= 1
        if level != self.map.level:
            reward += 0.8

        state = self.state_encode()

        return state, reward, status_code

    def auto_step(self):
        self.fire()
        status_code = self.move_to_enemy_and_update()

        if status_code:
            return status_code

        self.try_start_level()
        status_code = self.try_next_level()

        return status_code

    def _step(self, action):
        if action == 0:
            return
        keys = ["D", "S", "A", "W"]
        if action < 4:
            key_down(keys[action - 1])
        elif action == 5:
            key_down("D")
            key_down("S")
        elif action == 6:
            key_down("S")
            key_down("A")
        elif action == 7:
            key_down("A")
            key_down("W")
        elif action == 8:
            key_down("W")
            key_down("D")

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

        if self.boss is not None:
            state[17:20] = self.boss.state_encode()
        for i, enemy in enumerate(self.enemy_list):
            state[20 + i * 3:20 + i * 3 + 3] = enemy.state_encode()
        state[35:63] = self.bullets.state_encode()

        # print(state[0:7])
        # print(state[7:17])
        # for i in range(6):
        #     print(state[17+i*3:17+i*3+3])
        # for i in range(4):
        #     print(state[35+i*7:35+i*7+7])
        # print()

        return state

    def fire(self):
        if len(self.enemy_list) == 0 and self.boss is None:
            return -0.1

        target_list = self.enemy_list + [self.boss]
        direction = self.map.fire_direction(self.player, target_list)
        if direction is not None:
            fired = self.action_manager.fire_to(direction)
            return 0.1 + fired * 0.4
        return -0.1

    def try_start_level(self):
        def loop_act(direction):
            self.fire()
            self.action_manager.take_action_direction(direction)
            time.sleep(0.1)
            self.update()
            self.render()

        while self.new_level:
            if self.map.get_task() == 5:
                self.action_manager.take_action(0)
                time.sleep(2)
            elif self.map.map_index == 1:
                if self.map.level == 3:
                    self.action_manager.take_action(0)
                    time.sleep(2)
            elif self.map.map_index == 3:
                if self.map.level == 1:
                    direction = Direction.Down
                    self.action_manager.take_action_direction(direction)
                    time.sleep(2)
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
        return len(self.enemy_list) == 0 and self.boss is None and len(self.bullets) == 0

    def try_next_level(self):
        loop_start_time = time.time()
        while self.can_next_level() and not self.new_level:
            direction, path = self.map.get_next_direction(self.player, self.falling_items)
            self.action_manager.take_action_direction(direction)
            time.sleep(0.1)
            self.render(path)
            status_code = self.update()

            if status_code:
                return status_code
            if time.time() - loop_start_time > self.stay_time_out:
                return -3

        return 0

    def move_to_enemy_and_update(self):
        if self.can_next_level():
            return 0
        if self.new_level:
            return 0
        self.move_index = None
        self.move_end = None

        start_time = time.time()
        direction, path, self.move_index = self.map.get_shot_place_direction(self.player, self.move_index, self.move_end)
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

    def render(self, path=None):
        image = np.zeros((MAZE_HEIGHT, MAZE_WIDTH, 3), dtype=np.uint8)
        if self.map.debug_image is not None:
            image += self.map.debug_image

        cv2.circle(image, (self.player.x, self.player.y), 4, (0, 255, 0), -1)
        cv2.circle(image, (self.player.x, self.player.y), 1, (0, 0, 255), -1)
        for enemy in self.enemy_list + [self.boss]:
            if enemy is None:
                continue
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


if __name__ == "__main__":
    import sys
    import traceback

    try:
        epochs = 100
        env = Env(time_out=3)
        env.init_window()

        print("Start Loop")
        for i in range(epochs):
            state, status = env.reset()
            while status == 0:
                status = env.auto_step()
                if status:
                    break
            time.sleep(1)
            env.game_window.recover(status)
            time.sleep(1)

    except Exception as e:
        print(traceback.format_exc())
        if len(sys.argv) <= 1:
            raise e

    finally:
        if len(sys.argv) > 1:
            input("Press Enter to continue...")
