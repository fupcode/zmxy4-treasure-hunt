"""统一入口透传参数及交互选择，不实际操作游戏。"""

import contextlib
import io
import sys
import types
import unittest
from unittest.mock import Mock, patch

import main as entry
from zmxy4_treasure_hunt.voyage import cli


class EntryTests(unittest.TestCase):
    def run_entry(self, command, argv, responses=()):
        module_name = ("zmxy4_treasure_hunt.maze.cli" if command == "maze"
                       else "zmxy4_treasure_hunt.voyage.cli")
        module = types.ModuleType(module_name)
        module.main = Mock(return_value=17)
        with patch.dict(sys.modules, {module_name: module}), \
                patch("builtins.input", side_effect=responses), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main(argv), 17)
        return module.main

    def test_child_arguments_and_exit_status_pass_through(self):
        for command, args in (("maze", ["--interval", "0.1", "--debug"]),
                              ("voyage", ["5", "--debug", "--map-index", "2", "--level", "1"]),
                              ("voyage", ["--help"])):
            with self.subTest(command=command, args=args):
                self.run_entry(command, [command, *args]).assert_called_once_with(args)

    def test_interactive_voyage_retries_invalid_choice_and_count(self):
        self.run_entry("voyage", [], ["oops", "1", "-1", "abc", "0", "5"]).assert_called_once_with(["5"])

    def test_interactive_voyage_default_count(self):
        self.run_entry("voyage", [], ["航海", ""]).assert_called_once_with(["3"])

    def test_interactive_maze_needs_no_count(self):
        self.run_entry("maze", [], ["2"]).assert_called_once_with([])

    def test_help_and_unknown_command_do_not_prompt(self):
        for args, code in ((["--help"], 0), (["unknown"], 2)):
            with self.subTest(args=args), patch("builtins.input") as prompt, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    entry.main(args)
                self.assertEqual(error.exception.code, code)
                prompt.assert_not_called()

    def test_interactive_eof_exits(self):
        with patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(entry.main([]), 130)


class VoyageCliTests(unittest.TestCase):
    def test_help_and_invalid_count_do_not_load_environment(self):
        for args, code in ((["--help"], 0), (["0"], 2), (["-1"], 2)):
            with self.subTest(args=args), patch.dict(sys.modules, {"zmxy4_treasure_hunt.voyage.env": None}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli.main(args)
                self.assertEqual(error.exception.code, code)

    def test_voyage_target_and_read_only_options(self):
        env = types.ModuleType("zmxy4_treasure_hunt.voyage.env")
        env.Voyager = Mock()
        env.auto_loop = Mock()
        voyager = env.Voyager.return_value
        with patch.dict(sys.modules, {env.__name__: env}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["5"]), 0)
            env.auto_loop.assert_called_once_with(voyager, 5)
            voyager.init_window.assert_called_once_with(focus_game=True)
            voyager.reset_mock()
            env.auto_loop.reset_mock()
            voyager.debug_step.side_effect = [0, -3, 1]
            with patch("cv2.destroyWindow") as close:
                self.assertEqual(cli.main(["--debug", "--map-index", "2", "--level", "1"]), 0)
            voyager.init_window.assert_called_once_with(focus_game=False)
            voyager.map.reset.assert_called_once_with(2)
            voyager.map.next_level.assert_called_once_with()
            env.auto_loop.assert_not_called()
            close.assert_called_once_with("Monitor")
