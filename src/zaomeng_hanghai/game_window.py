import cv2
import os
import time
import numpy as np
import dxcam
import win32api
import win32con

from zaomeng_hanghai.core.onnx_model import OnnxModel, DetectionResult
from zaomeng_hanghai.core.action_manager import key_up, key_up_all


# 耗时记录装饰器
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' of GameWindow executed in {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


def set_cursor_pos(x, y):
    win32api.SetCursorPos((x, y))


def click(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


class GameWindow:
    def __init__(self, yolo_weight_path="weights/best.onnx", *, capture_enabled=True):
        self.camera = dxcam.create() if capture_enabled else None

        self.model = OnnxModel(yolo_weight_path)

        self.window_box = None
        self.window_size = None
        self.left_top_mask_rate = (0.078, 0.14)
        self.left_bottom_mask_rate = (0.326, 0.872)

        self.window_image = None
        self.hit_point = None
        self.map_index = None

        self.player_box = None
        self.detections = {}
        self.enemy_boxes = None
        self.bullet_list_normal = None
        self.falling_items_normal = None

        self.items = {"pao": False, "dan": False, "fan": False}

    def init_window(self, outer_window_box=None):
        screen_image = self.capture_screen(outer_window_box)
        screen_image_hls = cv2.cvtColor(screen_image, cv2.COLOR_RGB2HLS)
        self.window_box = self.locate_window(screen_image_hls)

        if outer_window_box is not None:
            self.window_box[0] += outer_window_box[0]
            self.window_box[1] += outer_window_box[1]
            self.window_box[2] += outer_window_box[0]
            self.window_box[3] += outer_window_box[1]
        width = self.window_box[2] - self.window_box[0]
        height = self.window_box[3] - self.window_box[1]
        self.window_size = (width, height)

        print("Window Box:", self.window_box)
        print(f"Window Size: {width} x {height} = {width * height}")

    def get_map_index_at_start(self):
        self.window_image = self.capture_screen(box=self.window_box)
        self.map_index = self.get_map_index(self.window_image)

        return self.map_index

    def update(self):
        return self.update_from_image(self.capture_screen(box=self.window_box))

    def update_from_image(self, image, *, conf_threshold=0.4, check_status=True):
        """处理已裁剪为游戏窗口的 RGB 图片，与实时更新共用感知流程。"""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image with shape (H, W, 3)")
        self.window_image = image.copy()
        self.detections = {}

        self.detect_item(self.window_image)

        mask_width = int(round(self.window_image.shape[1] * self.left_top_mask_rate[0]))
        mask_height = int(round(self.window_image.shape[0] * self.left_top_mask_rate[1]))
        self.window_image[0:mask_height, 0:mask_width] = 255
        mask_width = int(round(self.window_image.shape[1] * self.left_bottom_mask_rate[0]))
        mask_height = int(round(self.window_image.shape[0] * self.left_bottom_mask_rate[1]))
        self.window_image[mask_height:, 0:mask_width] = 255

        map_contours = self.locate_map_contours(self.window_image)
        self.hit_point = self.get_hit_point(self.window_image)

        if check_status and (len(map_contours) != 1 or self.hit_point == 0):
            if self.is_failed(self.window_image):
                return -2
            if self.is_successful(self.window_image):
                return 1
            return -1

        det = self.model(self.window_image, conf_threshold=conf_threshold)
        self.detections = det

        self.player_box = self.filter_player(det)
        self.enemy_boxes = det['enemy']
        self.falling_items_normal = self.filter_falling_items(det)
        self.bullet_list_normal = self.filter_bullet(det)

        return 0

    def set_cursor_pos_normal(self, xn, yn):
        x = int(round(self.window_box[0] + self.window_size[0] * xn))
        y = int(round(self.window_box[1] + self.window_size[1] * yn))
        set_cursor_pos(x, y)

    def click_normal(self, xn, yn):
        x = int(round(self.window_box[0] + self.window_size[0] * xn))
        y = int(round(self.window_box[1] + self.window_size[1] * yn))
        click(x, y)

    def recover(self, status, end=False):
        time.sleep(3)
        key_up("J")
        key_up_all()
        if status == -2:
            print("Failed.")
            self.click_normal(0.4, 0.712)
            time.sleep(3)
        elif status in (-3, -4):
            print("Combat Time Out." if status == -3 else "Engine Time Out.")
            self.click_normal(0.8, 0.08)
            time.sleep(3)
            self.start_hanghai()

        elif status == 1:
            print("Successful.")
            self.click_normal(0.784, 0.765)
            time.sleep(3)
            self.start_hanghai(end=end)

    def start_hanghai(self, end=False):
        max_confidence = 0
        icon_box: DetectionResult | None = None
        for i in range(10):
            self.set_cursor_pos_normal(0.75, 0.072)
            image = self.capture_screen(self.window_box)
            det = self.model(image)
            icon_boxes = det['player']

            for icon in icon_boxes:
                if icon.conf > max_confidence:
                    max_confidence = icon.conf
                    icon_box = icon

            if icon_box is not None:
                break

            time.sleep(1)

        if icon_box is None:
            raise TimeoutError("Can't find the icon.")

        self.click_normal(icon_box.xywhn[0], icon_box.xywhn[1])

        time.sleep(0.5)
        self.click_normal(0.352, 0.706)
        if not end:
            time.sleep(0.2)
            self.click_normal(0.352, 0.706)

        time.sleep(3)

    def is_successful(self, image):
        success_box_rate = (0.567, 0.772, 0.434, 0.684)

        success_image = self.crop_by_rate(image, success_box_rate)
        success_mask = cv2.inRange(success_image, np.array([220, 220, 220]), np.array([255, 255, 255]))
        char_num = self.get_filtered_bounding_box_count(success_mask, min_area=10)

        if char_num > 30:
            return True
        return False

    def is_failed(self, image):
        fail_box_rate = (0.06, 0.18, 0.18, 0.72)

        fail_image = self.crop_by_rate(image, fail_box_rate)
        fail_mask = cv2.inRange(fail_image, np.array([50, 50, 50]), np.array([100, 100, 100]))
        if np.mean(fail_mask) > 200:
            return True
        return False

    def filter_player(self, det):
        player_list = det['player']
        if len(player_list) == 0:
            return self.player_box

        if len(player_list) == 1:
            return player_list[0]

        max_confidence = 0
        player = None
        for player_box in player_list:
            if player_box.conf > max_confidence:
                max_confidence = player_box.conf
                player = player_box

        return player

    def filter_falling_items(self, det):
        falling_items = []
        for key in ['dan', 'fan', 'jia', 'pao']:
            if key == 'jia':
                if self.hit_point == 4:
                    continue
            elif self.items[key]:
                continue
            for box in det[key]:
                xywhn = box.xywhn
                falling_items.append((xywhn[0], xywhn[1]))

        return falling_items

    def filter_bullet(self, det):
        window_image_hls = np.zeros(self.window_image.shape, np.uint8)
        height, width = self.window_image.shape[:2]
        for bullet_box in det['bullet']:
            x1, y1, x2, y2 = bullet_box.xyxyn
            crop_box = (int(round(x1 * width)), int(round(y1 * height)),
                        int(round(x2 * width)), int(round(y2 * height)))

            image = self.crop_image_by_box(self.window_image, crop_box)

            if image is None or image.size == 0:
                continue

            hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            window_image_hls[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = hls
        bullet_list_normal = self.detect_bullet_normal(window_image_hls)

        return bullet_list_normal

    def detect_bullet_normal(self, hls_image):
        height, width, _ = hls_image.shape

        fire_lower = np.array([6, 60, 80])
        fire_upper = np.array([12, 220, 220])
        fire_mask = cv2.inRange(hls_image, fire_lower, fire_upper)

        head_lower = np.array([12, 70, 5])
        head_upper = np.array([32, 120, 30])
        head_mask = cv2.inRange(hls_image, head_lower, head_upper)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        head_contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = height * width * 0.0002  # Window Area 1948772
        bullet_list = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < min_area:
                continue

            r = self.get_centroid_and_direction(contour, head_contours)
            if r is None:
                continue

            x, y, d = r
            xn = x / width
            yn = y / height
            bullet_list.append((xn, yn, d))

        return bullet_list

    def detect_item(self, window_image):
        pao_rate = (0.0437, 0.0618, 0.906, 0.938)
        dan_rate = (0.121, 0.139, 0.904, 0.939)
        fan_rate = (0.198, 0.216, 0.906, 0.941)

        pao_image = self.crop_by_rate(window_image, pao_rate)
        dan_image = self.crop_by_rate(window_image, dan_rate)
        fan_image = self.crop_by_rate(window_image, fan_rate)

        pao_hls = cv2.cvtColor(pao_image, cv2.COLOR_RGB2HLS)
        dan_hls = cv2.cvtColor(dan_image, cv2.COLOR_RGB2HLS)
        fan_hls = cv2.cvtColor(fan_image, cv2.COLOR_RGB2HLS)

        lower = np.array([65, 15, 20])
        upper = np.array([100, 35, 150])
        pao_mask = cv2.inRange(pao_hls, lower, upper)
        dan_mask = cv2.inRange(dan_hls, lower, upper)
        fan_mask = cv2.inRange(fan_hls, lower, upper)

        self.items["pao"] = np.mean(pao_mask) < 100
        self.items["dan"] = np.mean(dan_mask) < 100
        self.items["fan"] = np.mean(fan_mask) < 100

    def capture_screen(self, box=None):
        if self.camera is None:
            raise RuntimeError("Screen capture is disabled; use update_from_image()")
        while True:
            image = self.camera.grab(region=box)

            if image is not None:
                return image

            time.sleep(0.001)

    def locate_window(self, hls_image):
        # 二值化
        wall_lower = np.array([0, 20, 0])
        wall_upper = np.array([105, 75, 255])
        wall_mask = cv2.inRange(hls_image, wall_lower, wall_upper)

        water_lower = np.array([90, 86, 100])
        water_upper = np.array([105, 200, 255])
        water_mask = cv2.inRange(hls_image, water_lower, water_upper)

        mask = cv2.bitwise_or(wall_mask, water_mask)

        # 膨胀
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        box = self.get_window_box(mask)
        return box

    def get_hit_point(self, window_image):
        rate = (0.09, 0.21, 0.038, 0.065)
        health_image = self.crop_by_rate(window_image, rate)

        # 二值化
        health_lower = np.array([150, 0, 0])
        health_upper = np.array([255, 200, 80])
        health_mask = cv2.inRange(health_image, health_lower, health_upper)

        min_area = health_image.shape[0] * health_image.shape[1] * 0.0005
        hit_point = self.get_filtered_bounding_box_count(health_mask, min_area=min_area)
        return hit_point

    def locate_map_contours(self, window_image):
        xr0 = 0.90
        yr0 = 0.055
        rate = (xr0, 0.975, yr0, 0.155)
        map_image = self.crop_by_rate(window_image, rate)

        # 转换为灰度图像
        gray = cv2.cvtColor(map_image, cv2.COLOR_RGB2GRAY)

        # 使用阈值处理，提取白色区域
        _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def get_map_index(self, window_image):
        xr0 = 0.90
        yr0 = 0.055
        contours = self.locate_map_contours(window_image)

        # 判断白色区域数量
        if len(contours) != 1:
            raise GameWindowError(f"Expected 1 white block in map area, but found {len(contours)}.")

        # 获取唯一白色区域的坐标
        x, y, _, _ = cv2.boundingRect(contours[0])
        height, width, _ = window_image.shape

        xr = (xr0 * width + x) / width
        yr = (yr0 * height + y) / height

        map_list = [
            (0.940, 0.0776),
            (0.914, 0.108),
            (0.931, 0.098),
            (0.931, 0.108),
            (0.948, 0.118),
            (0.931, 0.118),
            (0.922, 0.0776)
        ]
        map_index = self.find_nearest_point((xr, yr), map_list)

        return map_index

    def crop_image_by_box(self, image, box):
        x0, y0, x1, y1 = box
        return image[y0:y1, x0:x1]

    def crop_by_rate(self, image, rate):
        h, w, _ = image.shape
        return image[int(round(h * rate[2])):int(round(h * rate[3])),
                     int(round(w * rate[0])):int(round(w * rate[1]))]

    def get_centroid_and_direction(self, outer, inners):
        outer_x, outer_y, outer_w, outer_h = cv2.boundingRect(outer)
        outer_x2 = outer_x + outer_w
        outer_y2 = outer_y + outer_h
        for contour in inners:
            x, y, w, h = cv2.boundingRect(contour)
            xc = x + w / 2
            yc = y + h / 2

            if cv2.pointPolygonTest(outer, (xc, yc), False) < 0:
                continue

            if not self.is_point_in_box((xc, yc), (outer_x, outer_y, outer_x2, outer_y2)):
                continue

            angle = np.arctan2(yc - outer_y - outer_h / 2, xc - outer_x - outer_w / 2)
            angle = angle * 180 / np.pi
            if -45 <= angle <= 45:
                return xc, yc, 0
            elif 45 < angle <= 135:
                return xc, yc, 1
            elif -135 <= angle < -45:
                return xc, yc, 3
            else:
                return xc, yc, 2

        return None

    def get_filtered_bounding_box_list(self, mask, min_area=500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= min_area:
                bounding_boxes.append((x, y, w, h))

        return bounding_boxes

    def get_filtered_bounding_box_count(self, mask, min_area=100):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= min_area:
                count += 1

        return count

    def get_window_box(self, mask, min_size=(200, 200), aspect_ratio_range=(1.55, 1.65)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_box = None
        max_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w >= min_size[0] and h >= min_size[1]:
                aspect_ratio = w / h if h != 0 else 0
                if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    area = w * h
                    if area > max_area:
                        max_area = area
                        largest_box = (x, y, w, h)
        if largest_box is None:
            raise GameWindowError("找不到游戏窗口，请确保游戏已进入航海界面，且没有遮挡")
        return [largest_box[0], largest_box[1], largest_box[0] + largest_box[2], largest_box[1] + largest_box[3]]

    def find_nearest_point(self, point, point_list, tolerance=0.1):
        min_distance = float('inf')
        nearest_index = -1

        for i, (x, y) in enumerate(point_list):
            if abs(point[0] - x) > tolerance or abs(point[1] - y) > tolerance:
                raise GameWindowError(f"Point ({x}, {y}) at index {i} is too far from target point {point}.")

            distance = np.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2)

            if distance < min_distance:
                min_distance = distance
                nearest_index = i

        return nearest_index

    def is_point_in_box(self, point, box):
        x, y = point
        x1, y1, x2, y2 = box

        # 检查点是否在 box 的范围内
        return x1 <= x <= x2 and y1 <= y <= y2


class GameWindowError(Exception):
    pass
