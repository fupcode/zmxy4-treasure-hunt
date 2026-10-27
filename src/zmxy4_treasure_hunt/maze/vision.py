"""纯图像识别：输入 BGR 数组，不截图、不发送按键、不创建窗口。"""

from dataclasses import dataclass

import cv2
import numpy as np

from .viewport import locate_viewport


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
    valid_pixels: np.ndarray | None = None


def locate_character(image, valid_pixels=None):
    mask = cv2.inRange(image, (0, 0, 0), (0, 0, 0))
    if valid_pixels is not None:
        mask[~valid_pixels] = 0
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
    """用金色外框、矩形四角和暗色直边确定迷宫视区。"""
    try:
        match = locate_viewport(image)
    except ValueError as exc:
        raise DetectionError(str(exc)) from exc
    left, top, right, bottom = match.box
    return (left, top), (right, bottom)


def detect_walls(image, character, valid_pixels=None):
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
    if valid_pixels is not None:
        wall_mask[~valid_pixels] = 0

    # 去噪点
    hls = cv2.medianBlur(hls, 5)

    # 宝箱
    bao_mask = (hls[:, :, 0] <= 12) & (hls[:, :, 2] >= 120) & (hls[:, :, 2] <= 200)
    bao_mask = bao_mask.astype(np.uint8) * 255
    box_mask = bao_mask & ~character_mask
    if valid_pixels is not None:
        box_mask[~valid_pixels] = 0

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
    if valid_pixels is not None:
        # 可信区边界本身也会产生 Canny 边缘，收缩一圈后排除。
        safe = cv2.erode(valid_pixels.astype(np.uint8), kernel) != 0
        edges[~safe] = 0
        wall_dilated[~valid_pixels] = 0
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

    return WallObservation(wall_dilated, treasures, vertical_lines, horizontal_lines, valid_pixels)
