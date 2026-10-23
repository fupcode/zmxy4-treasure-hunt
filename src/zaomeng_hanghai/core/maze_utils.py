import numpy as np
from collections import deque
import heapq
import time
import math


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapse_time = time.time() - start_time
        if elapse_time > 0.03:
            print(f"Function '{func.__name__}' of maze_utils executed in {elapse_time:.4f} seconds.")
        return result

    return wrapper


def calculate_distance_from_obstacles(maze):
    """
    计算迷宫中每个通路点到最近障碍物的距离图。

    参数:
    - maze: 二维布尔数组（bool numpy array），0 表示通路，1 表示障碍。

    返回:
    - distance_from_obstacle: 二维数组，每个通路点到最近障碍物的距离，障碍物位置为 0。
    """
    rows, cols = maze.shape
    distance_from_obstacle = np.full((rows, cols), np.inf)

    # 使用队列初始化 BFS
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 1:
                distance_from_obstacle[r, c] = 0
                queue.append((c, r))  # 将所有障碍物点加入队列

    # 通过 BFS 计算每个点到最近障碍物的最短距离
    while queue:
        c, r = queue.popleft()
        current_dist = distance_from_obstacle[r, c]

        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 四个方向
            nc, nr = c + dc, r + dr
            if 0 <= nr < rows and 0 <= nc < cols and distance_from_obstacle[nr, nc] == np.inf:
                distance_from_obstacle[nr, nc] = current_dist + 1
                queue.append((nc, nr))

    return distance_from_obstacle


def calculate_direction_penalty(current, prev, start, stride, max_penalty=10):
    """
    计算方向惩罚，越靠近起点，惩罚越大，方向差异越大，惩罚越大。
    """
    cx, cy = current
    px, py = prev
    sx, sy = start

    # 计算当前方向和前一个方向的向量差异
    direction_current = (cx - px, cy - py)
    direction_prev = (px - sx, py - sy)

    # 计算方向差异的叉积（方向差异）
    cross_product = direction_current[0] * direction_prev[1] - direction_current[1] * direction_prev[0]
    direction_diff = abs(cross_product)

    # 计算从当前点到起点的曼哈顿距离（或欧几里得距离）
    distance_to_start = math.hypot(cx - sx, cy - sy)

    # 计算惩罚，距离越近惩罚越大，方向差异越大惩罚越大
    penalty = min(direction_diff / stride, max_penalty) * (1 / (distance_to_start + 1))

    return penalty


def find_nearest_passable_point(maze, start):
    rows, cols = maze.shape
    x, y = start
    if maze[y, x] == 0:
        return start  # 起点已经是通路

    queue = deque([start])
    visited = {start}

    while queue:
        cx, cy = queue.popleft()

        for dc, dr in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 四个方向
            nx, ny = cx + dc, cy + dr
            if 0 <= ny < rows and 0 <= nx < cols and (nx, ny) not in visited:
                if maze[ny, nx] == 0:
                    return (nx, ny)  # 找到最近的通路点
                visited.add((nx, ny))
                queue.append((nx, ny))

    return start  # 如果没有找到，返回原点（极端情况下）


def find_nearest_passable_path(maze, start, flee_region, stride=5, distance_maze=None, hate_factor=1, last_move_end=None):
    heuristic = lambda x, y: 0
    if last_move_end is not None:
        if flee_region[last_move_end[1], last_move_end[0]] == 0:
            # 启发函数
            heuristic = lambda x, y: abs(x - last_move_end[0]) + abs(y - last_move_end[1])

    # 迷宫的尺寸
    rows, cols = maze.shape

    # 八方向移动
    directions = [
        (0, stride), (stride, 0), (0, -stride), (-stride, 0),  # 上下左右
        (stride, stride), (stride, -stride), (-stride, stride), (-stride, -stride)  # 四个对角线
    ]

    # 计算每个点到障碍物的距离图
    if distance_maze is None:
        distance_from_obstacle = calculate_distance_from_obstacles(maze)
    else:
        distance_from_obstacle = distance_maze

    # 优先队列初始化
    open_set = []
    heapq.heappush(open_set, (0, start))

    # 路径跟踪字典
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        cx, cy = current

        # 判断是否进入终点附近的步幅半宽度范围
        if flee_region[cy, cx] == 0:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        # 八向扩展
        for dc, dr in directions:
            nx, ny = cx + dc, cy + dr

            if 0 <= ny < rows and 0 <= nx < cols and maze[ny, nx] == 0:
                new_cost = cost_so_far[current] + flee_region[ny, nx] + 1 / (distance_from_obstacle[ny, nx] + 1) * hate_factor

                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + heuristic(nx, ny)
                    heapq.heappush(open_set, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current

    return []


def find_path_to_point(maze, start, end, stride: int = 10, distance_maze=None, hate_factor=1):
    # 迷宫的尺寸
    rows, cols = maze.shape
    end = (int(end[0]), int(end[1]))

    # 八方向移动
    directions = [
        (0, stride), (stride, 0), (0, -stride), (-stride, 0),  # 上下左右
        (stride, stride), (stride, -stride), (-stride, stride), (-stride, -stride)  # 四个对角线
    ]

    # 计算每个点到障碍物的距离图
    if distance_maze is None:
        distance_from_obstacle = calculate_distance_from_obstacles(maze)
    else:
        distance_from_obstacle = distance_maze

    # 启发函数
    def heuristic(x, y):
        return abs(x - end[0]) + abs(y - end[1])

    # 优先队列初始化
    open_set = []
    heapq.heappush(open_set, (0, start))

    # 路径跟踪字典
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        cx, cy = current

        # 判断是否进入终点附近的步幅半宽度范围
        if abs(cx - end[0]) <= stride and abs(cy - end[1]) <= stride:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            path.append(end)
            return path

        # 八向扩展
        for dc, dr in directions:
            nx, ny = cx + dc, cy + dr

            if 0 <= ny < rows and 0 <= nx < cols and maze[ny, nx] == 0:
                new_cost = cost_so_far[current] + 1 + 1 / (distance_from_obstacle[ny, nx] + 1) * hate_factor

                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost + heuristic(nx, ny)
                    heapq.heappush(open_set, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current

    return [end]


def find_nearest_path_to_region(maze, start, flee_region, end_region, stride: int = 10):
    # 迷宫的尺寸
    rows, cols = maze.shape

    # 八方向移动
    directions = [
        (0, stride), (stride, 0), (0, -stride), (-stride, 0),  # 上下左右
        (stride, stride), (stride, -stride), (-stride, stride), (-stride, -stride)  # 四个对角线
    ]

    # 初始化队列
    queue = deque([start])

    # 路径跟踪字典
    came_from = {start: None}

    while queue:
        current = queue.popleft()
        cx, cy = current

        # 判断是否进入终点
        if end_region[cy, cx] > 0:
            # 回溯记录当前路径
            path = []
            temp = current
            while temp is not None:
                path.append(temp)
                temp = came_from[temp]
            path.reverse()
            return path

        # 八向扩展
        for dc, dr in directions:
            nx, ny = cx + dc, cy + dr

            if 0 <= ny < rows and 0 <= nx < cols and maze[ny, nx] == 0 and flee_region[ny, nx] == 0:
                neighbor = (nx, ny)
                if neighbor not in came_from:
                    queue.append(neighbor)
                    came_from[neighbor] = current

    return []


def find_best_path_to_region(maze, start, end_region, stride: int = 10, distance_maze=None, hate_factor=1, max_time=0.05):
    # 迷宫的尺寸
    start_time = time.time()
    rows, cols = maze.shape
    paths = []

    # 八方向移动
    directions = [
        (0, stride), (stride, 0), (0, -stride), (-stride, 0),  # 上下左右
        (stride, stride), (stride, -stride), (-stride, stride), (-stride, -stride)  # 四个对角线
    ]

    # 计算每个点到障碍物的距离图
    if distance_maze is None:
        distance_from_obstacle = calculate_distance_from_obstacles(maze)
    else:
        distance_from_obstacle = distance_maze

    # 优先队列初始化
    open_set = []
    heapq.heappush(open_set, (0, start))

    # 路径跟踪字典
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        cx, cy = current

        # 判断是否进入终点附近的步幅半宽度范围
        value = end_region[cy, cx]
        if value != 0:
            # 回溯记录当前路径
            path = []
            temp = current
            while temp in came_from:
                path.append(temp)
                temp = came_from[temp]
            path.reverse()
            paths.append((value, path))
            if value > 0.8:
                return path
            if time.time() - start_time > max_time:
                break

            continue

        # 八向扩展
        for dc, dr in directions:
            nx, ny = cx + dc, cy + dr

            if 0 <= ny < rows and 0 <= nx < cols and maze[ny, nx] == 0:
                new_cost = cost_so_far[current] + 1 + 1 / (distance_from_obstacle[ny, nx] + 1) * hate_factor

                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    priority = new_cost
                    heapq.heappush(open_set, (priority, (nx, ny)))
                    came_from[(nx, ny)] = current

    best_path = []
    best_value = -np.inf
    for value, path in paths:
        if value > best_value:
            best_value = value
            best_path = path

    return best_path
