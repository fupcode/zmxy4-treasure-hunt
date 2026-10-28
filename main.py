"""默认打开简单 GUI，带参数时保留命令行入口。"""

import argparse
import contextlib
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import traceback


def get_settings_path():
    base = Path(sys.executable) if getattr(sys, "frozen", False) else Path(__file__)
    return base.resolve().with_name("settings.json")


def load_settings(path):
    settings = {"mode": "maze", "voyage_count": 3, "minimize_on_start": True}
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(saved, dict):
        return settings
    if saved.get("mode") in ("maze", "voyage"):
        settings["mode"] = saved["mode"]
    count = saved.get("voyage_count")
    if type(count) is int and count > 0:
        settings["voyage_count"] = count
    if isinstance(saved.get("minimize_on_start"), bool):
        settings["minimize_on_start"] = saved["minimize_on_start"]
    return settings


def save_settings(path, settings):
    """在同目录写临时文件，写完后替换配置。"""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="settings-", suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            json.dump(settings, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class WindowNotFoundError(Exception):
    """启动时无法定位游戏窗口。"""


class RecognitionError(Exception):
    """运行中出现预期的窗口、画面识别或截图超时异常。"""


class LogWriter:
    def __init__(self, connection):
        self.connection = connection

    def write(self, text):
        if text:
            self.connection.send(("log", text))
        return len(text)

    def flush(self):
        pass


def release_keys():
    """释放操作按键；失败时返回错误提示，成功时返回 None。"""
    try:
        import win32api
        import win32con

        for char in "WASDJ":
            win32api.keybd_event(ord(char), 0, win32con.KEYEVENTF_KEYUP, 0)
    except Exception as exc:
        return f"释放操作按键失败：{exc}"


def run_task(mode, count):
    if mode == "maze":
        from zmxy4_treasure_hunt.maze.ft import FindTreasure
        from zmxy4_treasure_hunt.maze.vision.scene import DetectionError

        try:
            task = FindTreasure()
        except DetectionError as exc:
            raise WindowNotFoundError(
                "找不到迷宫窗口。请进入迷宫寻宝界面，将游戏放在主显示器上，"
                "开启最高画质并保持画面完整、无遮挡，然后重试。"
            ) from exc
        except TimeoutError as exc:
            raise RecognitionError(str(exc)) from exc
        try:
            task.main()
        except (DetectionError, TimeoutError) as exc:
            raise RecognitionError(str(exc)) from exc
    else:
        from zmxy4_treasure_hunt.voyage.env import Voyager, auto_loop
        from zmxy4_treasure_hunt.voyage.game_window import GameWindowError

        voyager = Voyager()
        try:
            try:
                voyager.init_window()
            except GameWindowError as exc:
                raise WindowNotFoundError(
                    "找不到航海窗口。请先进入出现船与海的航海界面，"
                    "开启最高画质并保持画面完整、无遮挡，然后重试。"
                ) from exc
            auto_loop(voyager, count)
        except (GameWindowError, TimeoutError) as exc:
            raise RecognitionError(str(exc)) from exc
        finally:
            if voyager.game_window.camera is not None:
                voyager.game_window.camera.release()


def task_worker(mode, count, connection):
    """子进程运行游戏任务，日志和结果通过管道交给界面。"""
    result = ("done", "已完成")
    writer = LogWriter(connection)
    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
        try:
            run_task(mode, count)
        except WindowNotFoundError as exc:
            result = ("window_error", str(exc))
        except RecognitionError as exc:
            result = ("error", str(exc))
        except BaseException as exc:
            traceback.print_exc()
            result = ("error", f"{type(exc).__name__}: {exc}")
        finally:
            if error := release_keys():
                result = ("error", error)
            connection.send(result)
            connection.close()


class Application:
    def __init__(self, root, settings_path=None):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.root = root
        self.settings_path = settings_path if settings_path is not None else get_settings_path()
        self.settings = load_settings(self.settings_path)
        self.process = self.connection = None
        self.result = None
        self.pipe_closed = False
        self.stopping = self.closing = False
        self.context = multiprocessing.get_context("spawn")
        root.title("造梦西游4寻宝助手")
        root.geometry("520x390")
        root.minsize(400, 300)
        icon = Path(__file__).with_name("treasure_hunt.ico")
        if icon.exists():
            root.iconbitmap(str(icon))
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        header = ttk.Frame(frame)
        header.pack(fill="x", pady=(0, 14))
        self.action_button = tk.Button(
            header, command=self.toggle_run, font=("Microsoft YaHei UI", 12, "bold"),
            width=10, padx=12, pady=12, relief="flat", borderwidth=0,
            foreground="white", activeforeground="white", disabledforeground="white",
            cursor="hand2",
        )
        self.action_button.pack(side="right", anchor="n", padx=(16, 0))
        self.update_action_button()
        options = ttk.Frame(header)
        options.pack(side="left", fill="x", expand=True)
        modes = ttk.Frame(options)
        modes.pack(anchor="w")
        self.mode = tk.StringVar(value=self.settings["mode"])
        self.count = tk.StringVar(value=str(self.settings["voyage_count"]))
        self.mode_buttons = []
        for label, mode in (("迷宫寻宝", "maze"), ("航海寻宝", "voyage")):
            button = ttk.Radiobutton(modes, text=label, value=mode,
                                     variable=self.mode, command=self.update_count)
            button.pack(side="left", padx=(0, 16))
            self.mode_buttons.append(button)
        count_options = ttk.Frame(options)
        count_options.pack(anchor="w", pady=(10, 0))
        ttk.Label(count_options, text="航海胜利次数").pack(side="left", padx=(0, 8))
        self.count_input = ttk.Spinbox(count_options, from_=1, to=9999, width=6, textvariable=self.count)
        self.count_input.pack(side="left")
        self.update_count()
        self.minimize_on_start = tk.BooleanVar(value=self.settings["minimize_on_start"])
        ttk.Checkbutton(options, text="寻宝时最小化窗口", variable=self.minimize_on_start).pack(
            anchor="w", pady=(10, 0))
        self.log = ScrolledText(frame, wrap="word", state="disabled", font=("Microsoft YaHei UI", 9))
        self.log.pack(fill="both", expand=True)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def update_count(self):
        self.count_input.configure(state="normal" if self.mode.get() == "voyage" else "disabled")

    def toggle_run(self):
        if self.process is None:
            self.start()
        else:
            self.stop()

    def update_action_button(self):
        running = self.process is not None
        self.action_button.configure(
            text="停止中…" if self.stopping and running else "停止" if running else "开始",
            background="#dc2626" if running else "#16a34a",
            activebackground="#b91c1c" if running else "#15803d",
            state="disabled" if self.stopping and running else "normal",
        )

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 1500:
            self.log.delete("1.0", f"{lines - 1500}.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        from tkinter import messagebox

        if self.process is not None:
            return
        try:
            count = int(self.count.get())
            if count <= 0:
                raise ValueError
        except ValueError:
            if self.mode.get() == "voyage":
                messagebox.showerror("次数无效", "胜利次数请输入正整数。", parent=self.root)
                return
            count = self.settings["voyage_count"]
        self.settings = {
            "mode": self.mode.get(),
            "voyage_count": count,
            "minimize_on_start": self.minimize_on_start.get(),
        }
        try:
            save_settings(self.settings_path, self.settings)
        except OSError as exc:
            self.append_log(f"配置保存失败：{exc}\n")
        self.result = None
        self.pipe_closed = False
        self.stopping = False
        sender = None
        try:
            self.connection, sender = self.context.Pipe(duplex=False)
            self.process = self.context.Process(target=task_worker, args=(self.mode.get(), count, sender))
            self.process.start()
        except Exception as exc:
            if self.connection is not None:
                self.connection.close()
            if self.process is not None:
                self.process.close()
            self.connection = self.process = None
            messagebox.showerror("启动失败", str(exc), parent=self.root)
            return
        finally:
            if sender is not None:
                sender.close()
        for button in self.mode_buttons:
            button.configure(state="disabled")
        self.count_input.configure(state="disabled")
        self.update_action_button()
        self.append_log("开始寻宝……\n")
        if self.minimize_on_start.get():
            self.root.iconify()

    def stop(self):
        if self.process is not None and not self.stopping:
            self.stopping = True
            self.update_action_button()
            # Windows 后台进程无法直接接收 Ctrl+C，结束后由主进程松键。
            if self.process.is_alive():
                self.process.terminate()

    def poll(self):
        if self.process is not None:
            # 终止后不读管道，避免读取被中断写入的消息。
            if not self.stopping:
                for _ in range(200):
                    try:
                        if not self.connection.poll():
                            break
                        kind, text = self.connection.recv()
                    except (EOFError, OSError):
                        self.pipe_closed = True
                        break
                    if kind == "log":
                        self.append_log(text)
                    else:
                        self.result = (kind, text)
            if not self.process.is_alive() and (self.stopping or self.pipe_closed):
                self.finish()
        if self.closing and self.process is None:
            self.root.destroy()
            return
        self.root.after(100, self.poll)

    def finish(self):
        from tkinter import messagebox

        self.process.join()
        exitcode = self.process.exitcode
        self.process.close()
        self.connection.close()
        self.connection = self.process = None
        result = ("done", "已停止") if self.stopping else (
            self.result or ("error", f"任务进程意外退出（退出码 {exitcode}）。"))
        if self.stopping or exitcode != 0:
            if error := release_keys():
                result = ("error", error)
        kind, text = result
        self.append_log(text + "\n")
        for button in self.mode_buttons:
            button.configure(state="normal")
        self.update_count()
        self.update_action_button()
        if not self.closing:
            self.root.deiconify()
            if kind in ("window_error", "error"):
                messagebox.showerror("未找到游戏窗口" if kind == "window_error" else "运行失败",
                                     text, parent=self.root)

    def close(self):
        self.closing = True
        self.stop()


def launch_gui():
    import tkinter as tk

    root = tk.Tk()
    Application(root)
    root.mainloop()
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return launch_gui()
    parser = argparse.ArgumentParser(
        description="航海寻宝与迷宫寻宝；不带参数时打开图形界面。",
        epilog="示例：main.py voyage 3；main.py maze --debug。子命令帮助：main.py maze --help。",
    )
    parser.add_argument("command", choices=("maze", "voyage"), help="maze：迷宫寻宝；voyage：航海寻宝")
    try:
        command = parser.parse_args(argv[:1]).command
        if command == "maze":
            from zmxy4_treasure_hunt.maze.cli import main as run
        else:
            from zmxy4_treasure_hunt.voyage.cli import main as run
        return run(argv[1:])
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")
        return 130


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
