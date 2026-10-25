"""Windows 短暂文件占用的重试与失败传播。"""

from pathlib import Path
import unittest
from unittest.mock import Mock, call, patch

from zaomeng_hanghai.core.file_io import replace_with_retry


def windows_error(code):
    error = PermissionError("file busy")
    error.winerror = code
    return error


class FileReplacementTests(unittest.TestCase):
    @patch("zaomeng_hanghai.core.file_io.time.sleep")
    def test_success_without_waiting(self, sleep):
        source, target = Mock(), Path("last.pt")
        replace_with_retry(source, target)
        source.replace.assert_called_once_with(target)
        sleep.assert_not_called()

    @patch("zaomeng_hanghai.core.file_io.time.sleep")
    def test_temporary_windows_conflicts_retry_then_succeed(self, sleep):
        for code in (5, 32, 33):
            with self.subTest(code=code):
                source = Mock()
                source.replace.side_effect = [windows_error(code), windows_error(code), None]
                sleep.reset_mock()
                replace_with_retry(source, Path("last.pt"))
                self.assertEqual(source.replace.call_count, 3)
                self.assertEqual(sleep.call_args_list, [call(0.05), call(0.1)])

    @patch("zaomeng_hanghai.core.file_io.time.sleep")
    def test_persistent_conflict_raises_and_keeps_source_and_target(self, sleep):
        source = Mock()
        error = windows_error(5)
        source.replace.side_effect = error
        with self.assertRaises(PermissionError) as raised:
            replace_with_retry(source, Path("last.pt"))
        self.assertIs(raised.exception, error)
        self.assertEqual(source.replace.call_count, 7)
        self.assertEqual(sleep.call_count, 6)
        source.unlink.assert_not_called()

    @patch("zaomeng_hanghai.core.file_io.time.sleep")
    def test_other_permission_errors_raise_immediately(self, sleep):
        source = Mock()
        source.replace.side_effect = PermissionError("not writable")
        with self.assertRaises(PermissionError):
            replace_with_retry(source, Path("last.pt"))
        source.replace.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
