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


def _match_by_distance(
    distances: list[list[int]],
    threshold_distance: int,
) -> list[int | None]:
    """为每个新对象分配旧对象索引，None 表示不匹配。"""
    assignment: list[int | None] = [None] * len(distances)
    best_assignment: list[int | None] = []
    best_match_count = -1
    best_total_distance = float("inf")

    def search(
        new_index: int,
        used_old: int,
        match_count: int,
        total_distance: int,
    ) -> None:
        nonlocal best_assignment
        nonlocal best_match_count, best_total_distance

        if new_index == len(distances):
            if (
                match_count > best_match_count
                or (
                    match_count == best_match_count
                    and total_distance < best_total_distance
                )
            ):
                best_match_count = match_count
                best_total_distance = total_distance
                best_assignment = assignment.copy()
            return

        # 选择一：不匹配当前新对象
        assignment[new_index] = None
        search(
            new_index + 1,
            used_old,
            match_count,
            total_distance,
        )

        # 选择二：匹配一个符合条件且尚未使用的旧对象
        for old_index, distance in enumerate(distances[new_index]):
            if distance > threshold_distance:
                continue

            if used_old & (1 << old_index):
                continue

            assignment[new_index] = old_index
            search(
                new_index + 1,
                used_old | (1 << old_index),
                match_count + 1,
                total_distance + distance,
            )

        assignment[new_index] = None

    search(0, 0, 0, 0)
    return best_assignment


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


class EnemyList(UserList[Enemy]):
    def __init__(self, max_disappear_count=1, threshold_distance=20):
        super().__init__()
        self.max_disappear_count = max_disappear_count
        self.threshold_distance = threshold_distance

    def update(self, enemy_box_list, maze) -> None:
        old_enemies = list(self.data)
        new_enemies = [Enemy(box, maze) for box in enemy_box_list]

        # 计算新旧敌人之间的曼哈顿距离
        distances = [
            [
                abs(new.x - old.x) + abs(new.y - old.y)
                for old in old_enemies
            ]
            for new in new_enemies
        ]

        best_assignment = _match_by_distance(
            distances,
            self.threshold_distance,
        )

        # 将匹配结果转换为旧敌人索引 -> 新敌人索引
        old_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(best_assignment)
            if old_index is not None
        }

        matched_new = {
            new_index
            for new_index, old_index in enumerate(best_assignment)
            if old_index is not None
        }

        updated_data = []

        # 更新旧敌人，保留短暂漏检的对象
        for old_index, old_enemy in enumerate(old_enemies):
            if old_index in old_to_new:
                new_enemy = new_enemies[old_to_new[old_index]]
                new_enemy.inherit(old_enemy)
                new_enemy.disappear_count = 0
                updated_data.append(new_enemy)
            else:
                old_enemy.disappear_count += 1
                if old_enemy.disappear_count <= self.max_disappear_count:
                    updated_data.append(old_enemy)

        # 未匹配的新敌人，直接作为新追踪对象加入
        for new_index, new_enemy in enumerate(new_enemies):
            if new_index not in matched_new:
                updated_data.append(new_enemy)

        self.data = updated_data

    def sort_by(self, player):
        # 根据与玩家的曼哈顿距离排序
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


class Bullets(UserList[Bullet]):
    def __init__(self, disappear_threshold=2):
        super().__init__()
        self.state_dim = 7 * 4
        self.data = []
        self.safe_bullets = []
        self.new_bullets = []

        self.disappear_threshold = disappear_threshold

    def update(self, bullet_list_normal, player) -> None:
        bullets = []
        safe_bullets = []

        for bullet_info in bullet_list_normal:
            bullet = Bullet(bullet_info)
            if self.is_safe(bullet, player):
                safe_bullets.append(bullet)
            else:
                bullets.append(bullet)

        # 优先处理靠近玩家的子弹
        bullets.sort(
            key=lambda bullet: abs(bullet.x - player.x)
                               + abs(bullet.y - player.y)
        )

        # 固定上一帧的子弹列表，避免匹配过程相互干扰
        old_bullets = list(self.data)
        matched_old = [False] * len(old_bullets)
        matched_new_for_old = {}
        new_bullets = []

        for bullet in bullets:
            candidates = []

            for i, old_bullet in enumerate(old_bullets):
                if matched_old[i]:
                    continue

                if not self._can_match(bullet, old_bullet):
                    continue

                distance = (
                        abs(bullet.x - old_bullet.x)
                        + abs(bullet.y - old_bullet.y)
                )
                candidates.append((distance, i))

            if not candidates:
                # 没有合适的旧子弹，视为新子弹
                new_bullets.append(bullet)
                continue

            # 选择距离最近的、尚未匹配的旧子弹
            _, i = min(candidates)

            matched_old[i] = True
            bullet.disappear_count = 0
            matched_new_for_old[i] = bullet

        # 更新旧子弹，保留短暂漏检的对象
        updated_data = []

        for i, old_bullet in enumerate(old_bullets):
            if matched_old[i]:
                updated_data.append(matched_new_for_old[i])
            else:
                old_bullet.disappear_count += 1

                if old_bullet.disappear_count <= self.disappear_threshold:
                    updated_data.append(old_bullet)

        # 添加本帧新出现的子弹
        updated_data.extend(new_bullets)

        self.data = updated_data
        self.safe_bullets = safe_bullets
        self.new_bullets = new_bullets

    def _can_match(self, bullet, old_bullet) -> bool:
        if bullet.d != old_bullet.d:
            return False

        dx = bullet.x - old_bullet.x
        dy = bullet.y - old_bullet.y

        if bullet.d == 0:
            return abs(dy) <= 5 and 0 <= dx <= FIRE_X_RANGE

        if bullet.d == 2:
            return abs(dy) <= 5 and 0 <= -dx <= FIRE_X_RANGE

        if bullet.d == 1:
            return abs(dx) <= 5 and 0 <= dy <= FIRE_Y_RANGE

        if bullet.d == 3:
            return abs(dx) <= 5 and 0 <= -dy <= FIRE_Y_RANGE

        return False

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

    def update(self, item_list_normal) -> None:
        old_items = list(self.data)
        new_items = [FallingItem(pos) for pos in item_list_normal]

        # 计算新旧物品之间的曼哈顿距离
        distances = [
            [
                abs(new.x - old.x) + abs(new.y - old.y)
                for old in old_items
            ]
            for new in new_items
        ]

        best_assignment = _match_by_distance(distances, self.threshold_distance)

        # 旧物品索引 -> 新物品索引
        old_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(best_assignment)
            if old_index is not None
        }

        matched_new = {
            new_index
            for new_index, old_index in enumerate(best_assignment)
            if old_index is not None
        }

        updated_data = []

        # 更新已有物品，保留暂时未检测到的物品
        for old_index, old_item in enumerate(old_items):
            if old_index in old_to_new:
                new_item = new_items[old_to_new[old_index]]
                new_item.disappear_count = 0
                updated_data.append(new_item)
            else:
                old_item.disappear_count += 1

                if old_item.disappear_count <= self.max_disappear_count:
                    updated_data.append(old_item)

        # 添加未匹配的新物品
        for new_index, new_item in enumerate(new_items):
            if new_index not in matched_new:
                updated_data.append(new_item)

        self.data = updated_data

    def sort_by(self, player):
        # 根据与玩家的曼哈顿距离排序
        self.data.sort(key=lambda item: abs(item.x - player.x) + abs(item.y - player.y))
