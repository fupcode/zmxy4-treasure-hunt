"""方向按键、开火冷却和临时转向管理。"""

import time

import win32api
import win32con

from .direction import (Direction, direction_list, is_reverse_direction, is_single_direction,
                        is_vertical_direction, is_horizontal_direction, get_horizontal_and_vertical_direction)


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

    def update_pao(self, pao):
        if self.pao == pao:
            return

        self.pao = pao
        if pao:
            self.fire_interval = 0.4
        else:
            self.fire_interval = 0.6

    def get_fire_direction_adjustment(self, fire_direction):
        if len(fire_direction.value) != 1:
            raise ValueError("Invalid fire direction.")
        if self.last_direction is None:
            return None, fire_direction

        # 处理单方向移动时
        if is_single_direction(self.last_direction):
            if fire_direction == self.last_direction:
                return None, None
            elif is_reverse_direction(self.last_direction, fire_direction):
                return None, fire_direction
            elif is_horizontal_direction(fire_direction):
                return None, fire_direction
            else:
                return self.last_direction, fire_direction

        # 处理双方向移动时
        if is_horizontal_direction(fire_direction):
            _, move_vertical = get_horizontal_and_vertical_direction(self.last_direction)
            if fire_direction == move_vertical:
                return None, None
            return None, fire_direction
        else:
            move_horizontal, move_vertical = get_horizontal_and_vertical_direction(self.last_direction)
            if fire_direction == move_vertical:
                return move_horizontal, None
            return move_horizontal, fire_direction

    def fire_to(self, direction):
        if self._can_fire():
            suspend_direction, temporary_direction = self.get_fire_direction_adjustment(direction)
            stop_to(suspend_direction)
            move_to(temporary_direction)
            time.sleep(0.002)
            key_down("J")
            key_up("J")
            stop_to(temporary_direction)
            move_to(suspend_direction)
            return 1
        return 0

    def modify_direction(self, direction):
        if self.last_direction is None:
            return None, direction
        if direction == self.last_direction:  # 完全符合
            return None, None
        # 处理部分冲突
        if not is_single_direction(self.last_direction):  # 双双部分冲突
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
