"""统一入口透传参数及 GUI 调度，不实际操作游戏。"""

import contextlib
import io
import sys
import types
import unittest
from unittest.mock import Mock, patch

import main as entry
from zmxy4_treasure_hunt.voyage import cli


class EntryTests(unittest.TestCase):
    def run_entry(self, command, argv):
        module_name = ("zmxy4_treasure_hunt.maze.cli" if command == "maze"
                       else "zmxy4_treasure_hunt.voyage.cli")
        module = types.ModuleType(module_name)
        module.main = Mock(return_value=17)
        with patch.dict(sys.modules, {module_name: module}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(argv), 17)
        return module.main

    def test_child_arguments_and_exit_status_pass_through(self):
        for command, args in (("maze", ["--interval", "0.1", "--debug"]),
                              ("voyage", ["5", "--debug", "--map-index", "2", "--level", "1"]),
                              ("voyage", ["--help"])):
            with self.subTest(command=command, args=args):
                self.run_entry(command, [command, *args]).assert_called_once_with(args)

    def test_no_arguments_launch_gui_without_prompt(self):
        with patch.object(entry, "launch_gui", return_value=0) as gui, patch("builtins.input") as prompt:
            self.assertEqual(entry.main([]), 0)
        gui.assert_called_once_with()
        prompt.assert_not_called()

    def test_help_and_unknown_command_do_not_prompt(self):
        for args, code in ((["--help"], 0), (["unknown"], 2)):
            with self.subTest(args=args), patch("builtins.input") as prompt, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    entry.main(args)
                self.assertEqual(error.exception.code, code)
                prompt.assert_not_called()

class VoyageCliTests(unittest.TestCase):
    def test_help_and_invalid_count_do_not_load_environment(self):
        for args, code in ((["--help"], 0), (["0"], 2), (["-1"], 2)):
            with self.subTest(args=args), patch.dict(sys.modules, {"zmxy4_treasure_hunt.voyage.env": None}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(args)
                self.assertEqual(error.exception.code, code)

    def test_voyage_target_and_read_only_options(self):
        for debug in (False, True):
            for read_only in (False, True):
                with self.subTest(debug=debug, read_only=read_only):
                    env = types.ModuleType("zmxy4_treasure_hunt.voyage.env")
                    env.Voyager = Mock()
                    env.auto_loop = Mock()
                    cv2 = types.ModuleType("cv2")
                    cv2.destroyWindow = Mock()
                    voyager = env.Voyager.return_value
                    voyager.debug_step.side_effect = [0, -3, 1]
                    args = ["5"]
                    if debug:
                        args.append("--debug")
                    if read_only:
                        args.extend(["--read-only", "--map-index", "2", "--level", "1"])
                    with patch.dict(sys.modules, {env.__name__: env, "cv2": cv2}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(cli.main(args), 0)
                    close = cv2.destroyWindow
                    env.Voyager.assert_called_once_with(render_enabled=debug)
                    voyager.init_window.assert_called_once_with(focus_game=not read_only)
                    if read_only:
                        voyager.map.reset.assert_called_once_with(2)
                        voyager.map.next_level.assert_called_once_with()
                        self.assertEqual(voyager.debug_step.call_count, 3)
                        env.auto_loop.assert_not_called()
                    else:
                        env.auto_loop.assert_called_once_with(voyager, 5)
                        voyager.debug_step.assert_not_called()
                    if read_only and debug:
                        close.assert_called_once_with("Monitor")
                    else:
                        close.assert_not_called()
