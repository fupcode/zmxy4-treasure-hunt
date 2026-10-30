import unittest
from unittest.mock import Mock, patch

from zmxy4_treasure_hunt import window_permissions as permissions


class WindowPermissionTests(unittest.TestCase):
    def test_coordinate_resolves_top_level_window_process(self):
        with patch.object(permissions.win32gui, 'WindowFromPoint', return_value=10) as at, \
             patch.object(permissions.win32gui, 'GetAncestor', return_value=20) as ancestor, \
             patch.object(permissions.win32process, 'GetWindowThreadProcessId', return_value=(1, 42)), \
             patch.object(permissions, 'get_process_permissions') as query:
            self.assertIs(permissions.get_window_permissions((100, 200)), query.return_value)
        at.assert_called_once_with((100, 200))
        ancestor.assert_called_once_with(10, permissions.win32con.GA_ROOT)
        query.assert_called_once_with(42)

    def test_comparison_rejects_only_lower_integrity(self):
        for own, target in ((8192, 12288), (8192, 8192), (12288, 8192), (12288, 12288)):
            with self.subTest(own=own, target=target), \
                 patch.object(permissions, 'get_window_permissions', return_value=permissions.ProcessPermissions(1, True, target)), \
                 patch.object(permissions, 'get_process_permissions', return_value=permissions.ProcessPermissions(2, False, own)):
                if own < target:
                    with self.assertRaisesRegex(permissions.WindowPermissionError, '以管理员身份运行'):
                        permissions.ensure_window_permission((10, 20))
                else:
                    permissions.ensure_window_permission((10, 20))

    def test_query_failure_is_reported_without_assuming_lower_permission(self):
        with patch.object(permissions, 'get_window_permissions', side_effect=permissions.win32api.error(5, 'OpenProcess', 'Access denied')):
            with self.assertRaisesRegex(permissions.WindowPermissionError, '无法查询'):
                permissions.ensure_window_permission((10, 20))

    def test_token_query_failure_closes_both_handles(self):
        process, token = Mock(), Mock()
        with patch.object(permissions.win32api, 'OpenProcess', return_value=process), \
             patch.object(permissions.win32security, 'OpenProcessToken', return_value=token), \
             patch.object(permissions.win32security, 'GetTokenInformation', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                permissions.get_process_permissions(42)
        token.Close.assert_called_once_with()
        process.Close.assert_called_once_with()
