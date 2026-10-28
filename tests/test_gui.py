"""验证任务日志、错误提示与停止清理，不实际操作游戏。"""

import unittest
import time
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import main as entry


class SettingsTests(unittest.TestCase):
    def test_saved_settings_round_trip(self):
        settings = {"mode": "voyage", "voyage_count": 5, "minimize_on_start": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            entry.save_settings(path, settings)
            self.assertEqual(entry.load_settings(path), settings)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_missing_damaged_and_invalid_settings_use_defaults(self):
        defaults = {"mode": "maze", "voyage_count": 3, "minimize_on_start": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            self.assertEqual(entry.load_settings(path), defaults)
            for text in ('{', '[]', '{"mode":"other","voyage_count":false,"minimize_on_start":"false"}'):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    self.assertEqual(entry.load_settings(path), defaults)

    def test_frozen_settings_are_next_to_executable(self):
        with patch.object(entry.sys, "frozen", True, create=True), \
                patch.object(entry.sys, "executable", str(Path.cwd() / "dist" / "assistant.exe")):
            self.assertEqual(entry.get_settings_path(), Path.cwd() / "dist" / "settings.json")


def smoke_worker(mode, count, connection):
    """真实 spawn 子进程的替身，不导入游戏模块、不发送按键。"""
    if mode == "maze":
        time.sleep(30)
    else:
        connection.send(("log", f"胜利 {count}/{count}\n"))
        connection.send(("done", "已完成"))
        connection.close()


class WorkerTests(unittest.TestCase):
    def test_logs_and_completion_are_forwarded_and_keys_released(self):
        connection = Mock()
        with patch.object(entry, "run_task", side_effect=lambda *_: print("进入第 1 关")), \
                patch.object(entry, "release_keys", return_value=None) as release:
            entry.task_worker("voyage", 3, connection)
        release.assert_called_once_with()
        messages = [call.args[0] for call in connection.send.call_args_list]
        self.assertIn(("log", "进入第 1 关"), messages)
        self.assertEqual(messages[-1], ("done", "已完成"))
        connection.close.assert_called_once_with()

    def test_window_errors_are_distinct_from_other_errors(self):
        for error, kind in ((entry.WindowNotFoundError("请进入游戏"), "window_error"),
                            (entry.RecognitionError("未检测到迷宫墙壁线条"), "error"),
                            (RuntimeError("模型加载失败"), "error")):
            with self.subTest(kind=kind):
                connection = Mock()
                with patch.object(entry, "run_task", side_effect=error), \
                        patch.object(entry, "release_keys", return_value=None) as release:
                    entry.task_worker("maze", 3, connection)
                release.assert_called_once_with()
                self.assertEqual(connection.send.call_args.args[0][0], kind)
                logs = "".join(call.args[0][1] for call in connection.send.call_args_list
                               if call.args[0][0] == "log")
                if isinstance(error, (entry.WindowNotFoundError, entry.RecognitionError)):
                    self.assertNotIn("Traceback", logs)
                    self.assertEqual(connection.send.call_args.args[0][1], str(error))
                else:
                    self.assertIn("Traceback", logs)
                    self.assertIn("模型加载失败", logs)


class LifecycleTests(unittest.TestCase):
    def make_app(self):
        app = entry.Application.__new__(entry.Application)
        for name in ("root", "process", "connection"):
            setattr(app, name, Mock())
        app.mode_buttons = [Mock(), Mock()]
        app.update_count = Mock()
        app.update_action_button = Mock()
        app.append_log = Mock()
        app.result = ("done", "已完成")
        app.stopping = app.closing = app.pipe_closed = False
        return app

    def test_stop_does_not_read_interrupted_pipe_and_releases_keys(self):
        app = self.make_app()
        app.process.is_alive.side_effect = [True, False]
        process, connection = app.process, app.connection
        with patch.object(entry, "release_keys", return_value=None) as release:
            app.stop()
            app.poll()
        process.terminate.assert_called_once_with()
        process.join.assert_called_once_with()
        release.assert_called_once_with()
        connection.poll.assert_not_called()
        connection.close.assert_called_once_with()
        self.assertIsNone(app.process)
        app.append_log.assert_called_with("已停止\n")

    def test_completion_waits_for_remaining_pipe_messages(self):
        app = self.make_app()
        app.result = None
        app.process.is_alive.return_value = False
        # 本轮检查后进程恰好退出，下一轮才能读到最终结果。
        app.connection.poll.return_value = False
        app.poll()
        self.assertIsNotNone(app.process)
        app.connection.poll.return_value = True
        app.connection.recv.side_effect = [("done", "已完成"), EOFError]
        app.poll()
        self.assertIsNone(app.process)
        app.append_log.assert_called_with("已完成\n")

    def test_close_stops_task_before_destroying_window(self):
        app = self.make_app()
        app.process.is_alive.side_effect = [True, False]
        with patch.object(entry, "release_keys", return_value=None):
            app.close()
            app.root.destroy.assert_not_called()
            app.poll()
        app.root.destroy.assert_called_once_with()
        app.root.deiconify.assert_not_called()


class GuiSmokeTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk

        self.root = tk.Tk()
        self.root.withdraw()
        self.settings_directory = tempfile.TemporaryDirectory()
        self.app = entry.Application(self.root, Path(self.settings_directory.name) / "settings.json")

    def tearDown(self):
        if self.app.process is not None:
            self.app.process.terminate()
            self.app.process.join()
            self.app.process.close()
            self.app.connection.close()
        self.root.destroy()
        self.settings_directory.cleanup()

    def wait_for_exit(self):
        deadline = time.monotonic() + 10
        while self.app.process is not None and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertIsNone(self.app.process, "任务未能及时退出")

    def test_real_spawn_log_completion_and_restart(self):
        self.assertEqual(self.app.mode.get(), "maze")
        self.assertTrue(self.app.count_input.instate(["disabled"]))
        self.app.mode.set("voyage")
        self.app.update_count()
        self.assertFalse(self.app.settings_path.exists())
        with patch.object(entry, "task_worker", smoke_worker), \
                patch.object(self.root, "iconify"), patch.object(self.root, "deiconify"):
            for _ in range(2):
                self.app.action_button.invoke()
                self.assertEqual(entry.load_settings(self.app.settings_path), {
                    "mode": "voyage", "voyage_count": 3, "minimize_on_start": True,
                })
                self.assertEqual(self.app.action_button.cget("text"), "停止")
                self.assertEqual(self.app.action_button.cget("background"), "#dc2626")
                self.wait_for_exit()
                self.assertEqual(self.app.action_button.cget("text"), "开始")
                self.assertEqual(self.app.action_button.cget("background"), "#16a34a")
        self.assertIn("胜利 3/3", self.app.log.get("1.0", "end"))

    def test_maze_disables_count_and_stops_real_process(self):
        self.app.mode.set("maze")
        self.app.update_count()
        self.assertTrue(self.app.count_input.instate(["disabled"]))
        with patch.object(entry, "task_worker", smoke_worker), \
                patch.object(entry, "release_keys", return_value=None) as release, \
                patch.object(self.root, "iconify"), patch.object(self.root, "deiconify"):
            self.app.action_button.invoke()
            self.app.action_button.invoke()
            self.assertEqual(str(self.app.action_button.cget("state")), "disabled")
            self.wait_for_exit()
            release.assert_called_once_with()
        self.assertEqual(self.app.action_button.cget("text"), "开始")
        self.assertIn("已停止", self.app.log.get("1.0", "end"))
