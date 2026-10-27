import win32api
import win32con
import win32gui
import tkinter as tk
import ctypes
from PIL import Image, ImageGrab, ImageTk


class ClickCapture:
    def __init__(self, message="请点击选中游戏窗口"):
        self.root = None
        self.click_position = None
        self.message = message
        self.region = None
        self.preview_region = None

    def capture_click(self, event):
        # 确认当前显示的范围，避免销毁遮罩后命中另一个窗口。
        if self.preview_region is not None:
            self.click_position = (event.x_root, event.y_root)
            self.region = self.preview_region
            self.root.quit()

    def exit_capture(self, event):
        self.root.quit()

    def window_at(self, point):
        """按 Z 顺序寻找遮罩下面的顶层窗口。"""
        windows = []

        def collect(hwnd, _):
            if (hwnd != self.overlay_hwnd and win32gui.IsWindowVisible(hwnd)
                    and not win32gui.IsIconic(hwnd)):
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                except win32gui.error:
                    return True  # 窗口可能刚好关闭。
                left, top, right, bottom = rect
                if left <= point[0] < right and top <= point[1] < bottom:
                    windows.append(rect)
            return True

        win32gui.EnumWindows(collect, None)
        return windows[0] if windows else None

    def update_preview(self):
        region = self.window_at(win32gui.GetCursorPos())
        if region != self.preview_region:
            self.preview_region = region
            self.canvas.delete("preview")
            if region is not None:
                left, top, right, bottom = region
                x1, y1 = left - self.origin[0], top - self.origin[1]
                x2, y2 = right - self.origin[0], bottom - self.origin[1]
                # 裁到虚拟桌面内；返回值仍保留窗口的完整屏幕坐标。
                crop = (max(0, x1), max(0, y1),
                        min(self.screenshot.width, x2), min(self.screenshot.height, y2))
                self.preview_image = ImageTk.PhotoImage(self.screenshot.crop(crop), master=self.root)
                self.canvas.create_image(crop[0], crop[1], anchor="nw",
                                         image=self.preview_image, tags="preview")
                self.canvas.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                                             outline="#00c8ff", width=4, tags="preview")
            self.canvas.tag_raise("hint")
        self.preview_job = self.root.after(30, self.update_preview)

    def start_capture(self):
        self.click_position = None
        self.region = None
        self.preview_region = None
        self.preview_job = None
        # Tk 和 Win32 使用同一套物理像素坐标，兼容多屏及不同缩放比例。
        set_dpi_context = ctypes.windll.user32.SetThreadDpiAwarenessContext
        set_dpi_context.argtypes = [ctypes.c_void_p]
        set_dpi_context.restype = ctypes.c_void_p
        previous_dpi_context = set_dpi_context(ctypes.c_void_p(-4))
        try:
            self.origin = (win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN),
                           win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN))
            # 先截图再创建遮罩，预览不会把自身识别为待选窗口。
            self.screenshot = ImageGrab.grab(all_screens=True)
            self.root = tk.Tk()
            self.root.overrideredirect(True)
            self.root.attributes('-topmost', True)
            self.root.geometry(f"{self.screenshot.width}x{self.screenshot.height}"
                               f"{self.origin[0]:+d}{self.origin[1]:+d}")
            self.canvas = tk.Canvas(self.root, highlightthickness=0, cursor="crosshair")
            self.canvas.pack(fill=tk.BOTH, expand=True)
            dimmed = Image.blend(self.screenshot.convert("RGB"),
                                Image.new("RGB", self.screenshot.size, "black"), 0.45)
            self.background_image = ImageTk.PhotoImage(dimmed, master=self.root)
            self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)
            # 提示放在主屏幕上，即使副屏在左侧也能看到。
            hint_x = win32api.GetSystemMetrics(win32con.SM_CXSCREEN) // 2 - self.origin[0]
            hint_y = 40 - self.origin[1]
            self.hint = self.canvas.create_text(hint_x, hint_y, anchor="n", text=self.message,
                                                font=('Microsoft YaHei', 14), fill="white", tags="hint")
            self.root.bind("<Button-1>", self.capture_click)
            self.root.bind("<Escape>", self.exit_capture)
            self.root.update_idletasks()
            self.overlay_hwnd = win32gui.GetAncestor(self.root.winfo_id(), win32con.GA_ROOT)
            # Tk 对负坐标的 geometry 解释为距右/下边缘，使用 Win32 明确定位。
            win32gui.SetWindowPos(self.overlay_hwnd, win32con.HWND_TOPMOST,
                                  *self.origin, self.screenshot.width, self.screenshot.height,
                                  win32con.SWP_SHOWWINDOW)
            self.root.focus_force()
            self.update_preview()
            self.root.mainloop()
        finally:
            if self.root is not None:
                if self.preview_job is not None:
                    self.root.after_cancel(self.preview_job)
                self.root.destroy()
                self.root = None
            if previous_dpi_context:
                set_dpi_context(previous_dpi_context)

        return self.region