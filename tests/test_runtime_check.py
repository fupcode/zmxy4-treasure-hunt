"""依赖检查失败应阻止发布，并保留原始缺失模块信息。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main as entry
from zmxy4_treasure_hunt import runtime_check


class RuntimeCheckTests(unittest.TestCase):
    def test_entry_checks_dependencies_without_opening_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch.object(runtime_check, "check_winrt_dependencies"), \
                    patch.object(entry, "launch_gui") as gui, \
                    patch.object(runtime_check.sys, "frozen", True, create=True):
                self.assertEqual(entry.main(["--check-runtime", str(report)]), 0)
            self.assertEqual(json.loads(report.read_text(encoding="utf-8")),
                             {"ok": True, "frozen": True})
            gui.assert_not_called()

    def test_missing_dependency_returns_failure_and_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            with patch.object(runtime_check.importlib, "import_module",
                              side_effect=ModuleNotFoundError("winrt._winrt")):
                self.assertEqual(runtime_check.write_runtime_report(report), 1)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(result["ok"])
            self.assertIn("ModuleNotFoundError: winrt._winrt", result["error"])
