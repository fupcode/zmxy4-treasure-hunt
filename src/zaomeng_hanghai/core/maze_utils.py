import numpy as np
from collections import deque
import heapq
import time
import math
from scipy.ndimage import distance_transform_cdt


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
    x = min(cols - 1, max(0, int(round(start[0]))))
    y = min(rows - 1, max(0, int(round(start[1]))))
    start = (x, y)
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


def _region_path(maze, start, targets, distance_maze, hate_factor,
                 danger=None, max_time=0.02, preferred_end=None, stride=1,
                 allow_partial=True, cost_map=None):
    """指定步长的 A*；跨步路径逐格检查，不额外检查斜向两侧。

    max_time 是找到首条路径后的择优预算；预算耗尽时仍先找到可达目标，
    防止较远的窄目标因超时而始终返回空路径。无法完全脱险时返回风险更低的位置。
    cost_map 是非负逐格额外成本，仅影响路径成本，不启用 danger 的脱险语义。
    """
    rows, cols = maze.shape
    targets = np.asarray(targets).copy()
    targets[maze != 0] = 0
    if not np.any(targets):
        return []
    goal_distance = distance_transform_cdt(targets == 0, metric="chessboard")
    if distance_maze is None:
        distance_maze = calculate_distance_from_obstacles(maze)
    min_step_cost = 1 + hate_factor / (np.max(distance_maze) + 1)
    started = time.perf_counter()
    costs = {start: 0.0}
    parents = {}
    remaining = float(goal_distance[start[1], start[0]])
    queue = [(remaining * min_step_cost, remaining, 0.0, start)]
    best = None
    best_value = -np.inf
    safer = start
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1))

    def path_to(point):
        path = []
        while point in parents:
            path.append(point)
            point = parents[point]
        return path[::-1]

    while queue:
        _, _, cost, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        cx, cy = current
        if danger is not None and danger[cy, cx] < danger[safer[1], safer[0]]:
            safer = current
        value = targets[cy, cx]
        if value > 0:
            if danger is not None:
                return path_to(current)
            if value > best_value:
                best, best_value = current, value
            if value > 0.8 or time.perf_counter() - started >= max_time:
                return path_to(best)
            # 允许穿过低分候选区域，继续寻找更合适的射击位置。
        if best is not None and time.perf_counter() - started >= max_time:
            return path_to(best)
        for dx, dy in directions:
            nx, ny = cx + dx * stride, cy + dy * stride
            if not (0 <= nx < cols and 0 <= ny < rows) or maze[ny, nx]:
                continue
            step_cost = 0.0
            blocked = False
            for offset in range(1, stride + 1):
                x, y = cx + dx * offset, cy + dy * offset
                if maze[y, x]:
                    blocked = True
                    break
                step_cost += 1 + hate_factor / (distance_maze[y, x] + 1)
                if danger is not None:
                    step_cost += danger[y, x]
                if cost_map is not None:
                    step_cost += cost_map[y, x]
            if blocked:
                continue
            new_cost = cost + step_cost
            neighbor = (nx, ny)
            if new_cost < costs.get(neighbor, np.inf):
                costs[neighbor] = new_cost
                parents[neighbor] = current
                # 旧终点仅用于相同代价时择优，不能盖过本次实际危险和距离。
                preference = 0.0
                if preferred_end is not None:
                    preference = 1e-6 * (abs(nx - preferred_end[0]) + abs(ny - preferred_end[1]))
                remaining = float(goal_distance[ny, nx])
                priority = new_cost + remaining * min_step_cost + preference
                # 浮点累加误差不应打破等价路径的距离择优，导致开阔地大面积扩展。
                heapq.heappush(queue, (round(priority, 9), remaining, new_cost, neighbor))
    if best is not None:
        return path_to(best)
    return path_to(safer) if danger is not None and allow_partial else []


def _coarse_to_fine_region_path(maze, start, targets, distance_maze, hate_factor, **kwargs):
    """先搜索 5 格网格，未找到目标时再以 1 格完整搜索。"""
    if not np.any((np.asarray(targets) > 0) & (maze == 0)):
        return []
    if distance_maze is None:
        distance_maze = calculate_distance_from_obstacles(maze)
    for stride in (5, 1):
        path = _region_path(
            maze, start, targets, distance_maze, hate_factor,
            stride=stride, allow_partial=stride == 1, **kwargs,
        )
        if path or (not maze[start[1], start[0]] and targets[start[1], start[0]] > 0):
            return path
    return []


def find_nearest_passable_path(maze, start, flee_region, distance_maze=None, hate_factor=1, last_move_end=None):
    """先以 5 格寻找脱险路径，失败后以 1 格重试。"""
    return _coarse_to_fine_region_path(maze, start, (flee_region == 0).astype(np.float32),
                              distance_maze, hate_factor, danger=flee_region,
                              preferred_end=last_move_end)


def find_path_to_point(maze, start, end, stride: int = 10, distance_maze=None, hate_factor=1):
    # 迷宫的尺寸
    rows, cols = maze.shape
    end = (min(cols - 1, max(0, int(round(end[0])))), min(rows - 1, max(0, int(round(end[1])))))

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


def find_best_path_to_region(maze, start, end_region, distance_maze=None, hate_factor=1,
                             max_time=0.02, cost_map=None):
    """按目标评分择优；先粗后细搜索，可传入与地图同形状的非负逐格额外成本。"""
    return _coarse_to_fine_region_path(maze, start, end_region, distance_maze,
                              hate_factor, max_time=max_time, cost_map=cost_map)
