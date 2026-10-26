import tkinter as tk
import win32gui
import traceback

from zaomeng_hanghai.env import Env, auto_loop
from zaomeng_hanghai.game_window import GameWindowError


class ClickCapture:
    def __init__(self, message="Click anywhere"):
        self.root = None
        self.click_position = None
        self.message = message

    def capture_click(self, event):
        self.click_position = (event.x, event.y)
        self.root.quit()

    def exit_capture(self, event):
        self.root.quit()

    def start_capture(self):
        self.click_position = None
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)

        # 在屏幕中央显示信息
        label = tk.Label(self.root, text=self.message, font=('Arial', 16), bg='white')
        label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.root.bind("<Button-1>", self.capture_click)
        self.root.bind("<Escape>", self.exit_capture)  # 绑定Esc键

        # 强制窗口获得焦点
        self.root.focus_force()

        self.root.mainloop()
        self.root.destroy()

        if self.click_position:  # 如果有点击位置，则获取窗口的句柄和区域
            hwnd = win32gui.WindowFromPoint(self.click_position)
            rect = win32gui.GetWindowRect(hwnd)
            region = (rect[0], rect[1], rect[2], rect[3])
            return region
        return None  # 如果用户按Esc退出，则不返回任何区域


def get_count_from_user():
    count = input(">> ").strip()

    if count == "":
        return 3
    else:
        if not count.isdigit():
            print("非法输入，默认为3")
            return 3
        else:
            return int(count)


def main():
    print("本程序需要以管理员权限运行，否则运行速度较慢，达不到实时响应")
    print("请将此黑框放在游戏窗口旁，确保游戏窗口没有遮挡")
    print("请用最高画质，进入航海界面，确保出现船与海")
    print("若需退出，请关闭本程序窗口")
    print("请输入需要胜利的次数（默认为3），按Enter后开始自动航海")
    need_success_count = get_count_from_user()

    env = Env()
    try:
        env.init_window()
    except GameWindowError as e:
        click_capture = ClickCapture("请点击游戏窗口（按 Esc 取消）")
        region = click_capture.start_capture()
        if region is None:
            print("未选择游戏窗口")
            input("按 Enter 退出...")
            exit(0)
        env.init_window(outer_window_box=region)

    auto_loop(env, need_success_count)
    input("自动航海完毕，按 Enter 退出...")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行异常: {e}")
        traceback.print_exc()
        input("按 Enter 退出...")
