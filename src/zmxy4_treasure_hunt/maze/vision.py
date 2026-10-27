"""纯图像识别：输入 BGR 数组，不截图、不发送按键、不创建窗口。"""

from dataclasses import dataclass

import cv2
import numpy as np


class DetectionError(ValueError):
    """画面中缺少可靠的定位特征。"""


@dataclass(frozen=True)
class Character:
    x: int
    y: int
    radius: int


@dataclass
class WallObservation:
    image: np.ndarray
    treasures: list[tuple[int, int]]
    vertical_lines: list[float]
    horizontal_lines: list[float]


def locate_character(image):
    mask = cv2.inRange(image, (0, 0, 0), (0, 0, 0))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    opening[:20, :] = opening[-20:, :] = 0
    opening[:, :20] = opening[:, -20:] = 0
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise DetectionError("未找到人物的黑色定位特征")
    # 保留原先合并黑色特征、求凸包的策略；用两倍半径覆盖人物遮挡区域。
    hull = cv2.convexHull(np.vstack(contours))
    moments = cv2.moments(hull)
    if moments["m00"] == 0:
        raise DetectionError("人物定位特征面积为零")
    x = round(moments["m10"] / moments["m00"])
    y = round(moments["m01"] / moments["m00"])
    _, radius = cv2.minEnclosingCircle(hull)
    return Character(x, y, max(1, int(np.ceil(radius * 2))))


def locate_game(image):
    # 定义颜色
    specific_color = np.array([1, 1, 1])
    mask = cv2.inRange(image, specific_color, specific_color)

    # 使用形态学操作去除噪点
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 寻找特征区域并获取中心点
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        if cv2.contourArea(contour) >= 24:
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            centers.append((center_x, center_y))

    # 挑选满足条件的中心点组合
    best_pair = None
    best_ratio_diff = float('inf')
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            delta_x = abs(centers[i][0] - centers[j][0])
            delta_y = abs(centers[i][1] - centers[j][1])
            if delta_x > 400 and delta_y > 200:
                ratio = delta_x / delta_y
                ratio_diff = abs(ratio - 1.78)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_pair = (centers[i], centers[j])

    # 确定矩形
    if best_pair is not None:
        x_coords = [best_pair[0][0], best_pair[1][0]]
        y_coords = [best_pair[0][1], best_pair[1][1]]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 考虑完整的特征区域边界
        for contour in contours:
            for point in best_pair:
                if cv2.pointPolygonTest(contour, point, False) >= 0:
                    x, y, w, h = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w)
                    y_max = max(y_max, y + h)

        dx = (x_max - x_min) // 36
        dy = (y_max - y_min) // 12

        return (x_min + dx, y_min + dy), (x_max - dx, y_max - dy)

    raise DetectionError("未找到游戏窗口，请确认完整游戏画面位于主显示器")


def detect_walls(image, character):
    center_x, center_y, radius = character.x, character.y, character.radius

    def is_approximately_equal(val, target, tolerance=0.001):
        return abs(val - target) < tolerance
    # 转换为HLS颜色空间
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # 人物的圆形区域
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    character_mask = cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # 墙壁
    wall_mask = (hls[:, :, 2] <= 30) & (image[:, :, 0] >= 6)
    wall_mask = wall_mask.astype(np.uint8) * 255

    wall_mask = wall_mask & ~character_mask

    # 去噪点
    hls = cv2.medianBlur(hls, 5)

    # 宝箱
    bao_mask = (hls[:, :, 0] <= 12) & (hls[:, :, 2] >= 120) & (hls[:, :, 2] <= 200)
    bao_mask = bao_mask.astype(np.uint8) * 255
    box_mask = bao_mask & ~character_mask

    contours, _ = cv2.findContours(box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    treasures = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        w_min = image.shape[1] * 0.006
        if w > w_min and h > w_min:
            treasures.append((x, y))
            d = radius
            cv2.rectangle(wall_mask, (x - d, y - d), (x + d, y + d), 0, -1)

    # 形态学连接墙壁
    kernel = np.ones((9, 9), np.uint8)
    wall_dilated = cv2.dilate(wall_mask, kernel, iterations=1)
    wall_closed = cv2.erode(wall_dilated, kernel, iterations=1)
    # Canny边缘检测
    edges = cv2.Canny(wall_closed, 50, 100, apertureSize=3)
    # 使用Hough Transform检测直线
    # 宝箱纹理会产生大量短直线，不能仅按人物半径设置支持长度。
    line_threshold = max(radius, 20, min(image.shape[:2]) // 8)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

    vertical_lines = []
    horizontal_lines = []
    if lines is None:
        raise DetectionError("未检测到迷宫墙壁线条")
    for line in lines:
        rho, theta = line[0]
        if is_approximately_equal(theta, 0) or is_approximately_equal(theta, np.pi):  # 垂直
            vertical_lines.append(float(rho))
        elif is_approximately_equal(theta, np.pi / 2):  # 水平
            horizontal_lines.append(float(rho))

    return WallObservation(wall_dilated, treasures, vertical_lines, horizontal_lines)
