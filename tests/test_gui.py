"""验证任务日志、错误提示与停止清理，不实际操作游戏。"""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import Mock, call, patch

import main as entry


class LocationRetryTests(unittest.TestCase):
    def test_success_does_not_open_selector(self):
        locate = Mock(return_value="located")
        with patch('zmxy4_treasure_hunt.click_capture.ClickCapture') as selector:
            self.assertEqual(entry.locate_with_retry(locate, ValueError), "located")
        locate.assert_called_once_with()
        selector.assert_not_called()

    def test_failure_retries_once_with_selected_outer_window(self):
        locate = Mock(side_effect=[ValueError("missing"), "located"])
        outer = (100, 200, 900, 800)
        with patch('zmxy4_treasure_hunt.click_capture.ClickCapture') as selector:
            selector.return_value.start_capture.return_value = outer
            self.assertEqual(entry.locate_with_retry(locate, ValueError), "located")
        self.assertEqual(locate.call_args_list, [call(), call(outer_window_box=outer)])
        selector.return_value.start_capture.assert_called_once_with()

    def test_second_failure_is_not_retried(self):
        locate = Mock(side_effect=ValueError("missing"))
        with patch('zmxy4_treasure_hunt.click_capture.ClickCapture') as selector:
            selector.return_value.start_capture.return_value = (0, 0, 900, 800)
            with self.assertRaises(ValueError):
                entry.locate_with_retry(locate, ValueError)
        self.assertEqual(locate.call_count, 2)
        selector.return_value.start_capture.assert_called_once_with()

    def test_cancel_does_not_retry(self):
        locate = Mock(side_effect=ValueError("missing"))
        with patch('zmxy4_treasure_hunt.click_capture.ClickCapture') as selector:
            selector.return_value.start_capture.return_value = None
            with self.assertRaisesRegex(entry.WindowNotFoundError, "已取消"):
                entry.locate_with_retry(locate, ValueError)
        locate.assert_called_once_with()


class SettingsTests(unittest.TestCase):
    def test_saved_settings_round_trip(self):
        settings = {"mode": "voyage", "voyage_count": 5, "minimize_on_start": False,
                    "stop_hotkey": "Alt+F9"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            entry.save_settings(path, settings)
            self.assertEqual(entry.load_settings(path), settings)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_missing_damaged_and_invalid_settings_use_defaults(self):
        defaults = {"mode": "maze", "voyage_count": 3, "minimize_on_start": True,
                    "stop_hotkey": "F8"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            self.assertEqual(entry.load_settings(path), defaults)
            for text in ('{', '[]', '{"mode":"other","voyage_count":false,"minimize_on_start":"false"}',
                         '{"stop_hotkey":"Ctrl+F8"}'):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    self.assertEqual(entry.load_settings(path), defaults)

    def test_frozen_settings_are_next_to_executable(self):
        with patch.object(entry.sys, "frozen", True, create=True), \
                patch.object(entry.sys, "executable", str(Path.cwd() / "dist" / "assistant.exe")):
            self.assertEqual(entry.get_settings_path(), Path.cwd() / "dist" / "settings.json")


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
        for name in ("root", "process", "connection", "hotkey_button"):
            setattr(app, name, Mock())
        app.mode_buttons = [Mock(), Mock()]
        app.update_count = Mock()
        app.update_action_button = Mock()
        app.append_log = Mock()
        app.result = ("done", "已完成")
        app.hotkey = None
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
