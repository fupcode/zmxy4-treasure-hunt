"""玩家、敌人、子弹和掉落物，以及跨帧跟踪集合。"""

from collections import UserList

import numpy as np

from . import maze_utils
from .constants import (
    EXIT_DOWN,
    EXIT_LEFT,
    EXIT_RIGHT,
    EXIT_UP,
    FIRE_X_RANGE,
    FIRE_Y_RANGE,
    MAZE_HEIGHT,
    MAZE_WIDTH,
)


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
