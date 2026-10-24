"""地图构建、关卡路线和移动/攻击位置决策。"""

import time

import numpy as np
from scipy.ndimage import binary_dilation

from . import maze_utils
from .direction import Direction, direction_list, move_list
from .constants import (
    COLLISION_DOWN,
    COLLISION_LEFT,
    COLLISION_RIGHT,
    COLLISION_UP,
    EDGE_DOWN,
    EDGE_LEFT,
    EDGE_RIGHT,
    EDGE_UP,
    END_DOWN,
    END_LEFT,
    END_RIGHT,
    END_UP,
    EXIT_DOWN,
    EXIT_LEFT,
    EXIT_RIGHT,
    EXIT_UP,
    FIRE_X_RANGE,
    FIRE_Y_RANGE,
    HIT_HEIGHT,
    HIT_WIDTH,
    MAP_DOWN,
    MAP_LEFT,
    MAP_RIGHT,
    MAP_UP,
    MAZE_HEIGHT,
    MAZE_WIDTH,
)


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
