"""热键配置与 Windows 消息线程测试，不发送实际键盘输入。"""

import unittest

from zmxy4_treasure_hunt.hotkey import StopHotkey, parse_hotkey


class HotkeyTests(unittest.TestCase):
    def test_supported_shortcuts(self):
        self.assertEqual(parse_hotkey("F8"), (0, 0x77))
        self.assertEqual(parse_hotkey("Alt+F9"), (1, 0x78))
        self.assertEqual(parse_hotkey("Shift+Q"), (4, ord("Q")))

    def test_invalid_reserved_and_automated_keys_are_rejected(self):
        for text in (None, "", "F12", "Ctrl+F8", "Ctrl+Alt+F9", "Ctrl+F12", "Win+F8", "Ctrl+Ctrl+Q", "W", "J"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_hotkey(text)

    def test_native_listener_conflict_signal_and_release(self):
        first, second = StopHotkey(), StopHotkey()
        try:
            first.start("Alt+Shift+F11")
            with self.assertRaises(OSError):
                second.start("Alt+Shift+F11")
            # 直接投递到自己的消息线程，不向桌面或其他程序发送按键。
            first._user32.PostThreadMessageW(first._thread_id, 0x0312, 1, 0)
            self.assertTrue(first.triggered.wait(2))
        finally:
            first.close()
            second.close()
        third = StopHotkey()
        try:
            third.start("Alt+Shift+F11")
            self.assertIsNone(third.error)
        finally:
            third.close()
