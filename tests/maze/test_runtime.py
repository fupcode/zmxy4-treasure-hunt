import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from zmxy4_treasure_hunt.maze.runtime.capture import ScreenCapture
from zmxy4_treasure_hunt.maze.navigation.templates import MapTemplate
from zmxy4_treasure_hunt.maze.runtime.controls import MovementController
from zmxy4_treasure_hunt.maze.navigation import MapAlignmentError, NavigationResult
from zmxy4_treasure_hunt.maze.ft import FindTreasure
from zmxy4_treasure_hunt.maze.vision import Character, DetectionError, WallObservation


class CaptureTests(unittest.TestCase):
    def test_retry_and_region_preserve_bgr_frame(self):
        camera = Mock()
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[..., 0] = 200
        camera.grab.side_effect = [None, frame]
        capture = ScreenCapture(camera)
        with patch('zmxy4_treasure_hunt.maze.runtime.capture.time.sleep'):
            self.assertIs(capture.grab((10, 20, 30, 30)), frame)
        camera.grab.assert_called_with(region=(10, 20, 30, 30), new_frame_only=False)

    def test_timeout_and_close_are_bounded_and_idempotent(self):
        camera = Mock()
        camera.grab.return_value = None
        capture = ScreenCapture(camera)
        with patch('zmxy4_treasure_hunt.maze.runtime.capture.time.monotonic', side_effect=[0, 5]):
            with self.assertRaises(TimeoutError):
                capture.grab()
        capture.close()
        capture.close()
        camera.release.assert_called_once()
        with self.assertRaises(RuntimeError):
            capture.grab()

    def test_winrt_fallback_keeps_bgr(self):
        camera = Mock()
        with patch('dxcam.create', side_effect=[RuntimeError('DXGI unavailable'), camera]) as create:
            capture = ScreenCapture()
        self.assertIs(capture.camera, camera)
        self.assertEqual(create.call_args_list, [call(output_color='BGR'), call(output_color='BGR', backend='winrt')])
        capture.close()


class ControllerTests(unittest.TestCase):
    def test_unchanged_keys_are_not_repeated(self):
        controller = MovementController()
        with patch('zmxy4_treasure_hunt.maze.runtime.controls.key_down') as down, patch('zmxy4_treasure_hunt.maze.runtime.controls.key_up') as up:
            controller.apply('WD')
            controller.apply('WD')
            self.assertEqual(down.call_count, 2)
            up.assert_not_called()
            controller.apply('D')
            up.assert_called_once_with('W')
            controller.stop()
        self.assertEqual(controller.pressed, set())

    def test_stop_attempts_every_key_even_when_one_release_fails(self):
        controller = MovementController()
        with patch('zmxy4_treasure_hunt.maze.runtime.controls.key_up', side_effect=[OSError('input failed'), None, None, None]) as up:
            with self.assertRaises(OSError):
                controller.stop()
        self.assertEqual(up.call_args_list, [call(char) for char in 'WASD'])

    def test_invalid_movements_send_no_events(self):
        controller = MovementController()
        with patch('zmxy4_treasure_hunt.maze.runtime.controls.key_down') as down:
            with self.assertRaises(ValueError):
                controller.apply('Q')
        down.assert_not_called()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        permission_patch = patch('zmxy4_treasure_hunt.maze.ft.ensure_window_permission')
        self.check_permission = permission_patch.start()
        self.addCleanup(permission_patch.stop)

    def test_permission_failure_closes_capture_without_clicking(self):
        runner, capture, controller = self.make_runner()
        self.check_permission.side_effect = PermissionError('权限不足')
        with self.assertRaises(PermissionError):
            runner.main()
        self.check_permission.assert_called_once_with((310, 220))
        controller.click.assert_not_called()
        capture.close.assert_called_once_with()

    def make_runner(self, debug=False):
        capture, controller = Mock(), Mock()
        cells = np.full((5, 7), 2, dtype=np.uint8)
        cells[1, 1:6] = [1, 3, 1, 3, 4]
        with patch('zmxy4_treasure_hunt.maze.ft.locate_game', return_value=((10, 20), (610, 420))):
            runner = FindTreasure(capture=capture, controller=controller,
                                  map_template=MapTemplate(1, '测试地图', cells), debug=debug)
        capture.reset_mock()
        return runner, capture, controller

    def test_outer_window_only_limits_initial_capture_and_offsets_detected_region(self):
        capture, controller = Mock(), Mock()
        outer = (100, 200, 1000, 900)
        with patch('zmxy4_treasure_hunt.maze.ft.locate_game', return_value=((10, 20), (610, 420))) as locate:
            runner = FindTreasure(capture=capture, controller=controller, outer_window_box=outer)
        capture.grab.assert_called_once_with(outer)
        locate.assert_called_once_with(capture.grab.return_value)
        self.assertEqual(runner.region, (110, 220, 710, 620))
        runner.close()

    def test_failed_initial_location_closes_capture_and_releases_keys(self):
        capture, controller = Mock(), Mock()
        with patch('zmxy4_treasure_hunt.maze.ft.locate_game', side_effect=DetectionError('no window')):
            with self.assertRaises(DetectionError):
                FindTreasure(capture=capture, controller=controller)
        controller.stop.assert_called_once()
        capture.close.assert_called_once()

    def test_step_runs_real_navigation_without_hardware_or_ui(self):
        runner, capture, controller = self.make_runner()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        capture.grab.return_value = frame
        observation = WallObservation(np.zeros((400, 600), dtype=np.uint8), [(120, 50)],
                                      [10, 20, 80, 90, 150], [10, 20, 80, 90, 150])
        view = runner.template.cells.copy()
        runner.maze.load_view(view, (1, 1))
        with patch('zmxy4_treasure_hunt.maze.ft.locate_character', return_value=Character(50, 50, 10)) as locate, \
             patch('zmxy4_treasure_hunt.maze.ft.detect_walls', return_value=observation) as detect, \
             patch.object(runner.maze.grid, 'update'), \
             patch.object(runner.maze.grid, 'analyze', return_value=(view, (1, 3), (0, 0))):
            result = runner.step()
        self.assertIs(locate.call_args.args[1], runner.mask_restorer.valid_pixels)
        self.assertIs(detect.call_args.args[0], locate.call_args.args[0])
        self.assertIs(detect.call_args.args[2], runner.mask_restorer.valid_pixels)
        self.assertIsNot(locate.call_args.args[0], frame)
        self.assertEqual(runner.maze.char_coord, (1, 3))
        self.assertEqual(result.movements, 'D')
        self.assertTrue(result.approaching_treasure)
        capture.grab.assert_called_once_with((10, 20, 610, 420))
        controller.apply.assert_called_once_with('D')
        self.assertEqual(list(runner.maze.move_history), ['D'])

    def test_failed_step_stops_before_propagating_error(self):
        runner, capture, controller = self.make_runner()
        capture.grab.side_effect = TimeoutError('no frame')
        with self.assertRaises(TimeoutError):
            runner.step()
        controller.stop.assert_called_once()
        controller.apply.assert_not_called()

    def test_main_error_closes_devices_and_windows(self):
        runner, capture, controller = self.make_runner(debug=True)
        with patch('zmxy4_treasure_hunt.maze.ft.cv2.namedWindow', side_effect=RuntimeError('no GUI')), \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.destroyAllWindows') as destroy:
            with self.assertRaises(RuntimeError):
                runner.main()
        controller.stop.assert_called_once()
        capture.close.assert_called_once()
        destroy.assert_called_once()
        runner.close()
        capture.close.assert_called_once()

    def test_keyboard_interrupt_also_releases_inputs(self):
        runner, capture, controller = self.make_runner()
        capture.grab.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            runner.step()
        controller.stop.assert_called_once()

    def test_fake_treasure_stops_then_waits_five_seconds_without_clicks(self):
        runner, capture, controller = self.make_runner()
        capture.grab.return_value = np.zeros((120, 600, 3), dtype=np.uint8)
        events = Mock()
        events.attach_mock(controller.stop, 'stop')
        with patch('zmxy4_treasure_hunt.maze.ft.time.sleep') as sleep:
            events.attach_mock(sleep, 'sleep')
            self.assertTrue(runner.next())
        capture.grab.assert_called_once_with((10, 20, 610, 140))
        self.assertEqual(events.mock_calls, [call.sleep(0.5), call.stop(), call.sleep(5), call.stop()])
        controller.click.assert_not_called()

    def test_fake_treasure_keeps_map_location_grid_and_movement_history(self):
        runner, _, _ = self.make_runner(debug=True)
        maze = runner.maze
        view = np.full((5, 7), 2, dtype=np.uint8)
        view[1, 1:6] = [1, 3, 1, 3, 4]
        maze.load_view(view, (1, 1))
        result = maze.load_view(view, (1, 3))
        maze.record_movements(result.movements)
        maze.grid.interval0x = 10
        before = maze.maze.copy()
        location = maze.char_coord, maze.view_start
        history = list(maze.move_history)
        with patch.object(runner, 'step', return_value=result), patch.object(runner, 'render'), \
             patch.object(runner, 'next', return_value=True), \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.waitKey', side_effect=[-1, 27]), \
             patch('zmxy4_treasure_hunt.maze.ft.time.sleep'):
            runner.monitor()
        self.assertIs(runner.maze, maze)
        before[1, 5] = 1
        np.testing.assert_array_equal(maze.maze, before)
        self.assertEqual((maze.char_coord, maze.view_start), location)
        self.assertEqual(maze.grid.interval0x, 10)
        self.assertEqual(list(maze.move_history), history)
        # 恢复画面后假宝箱已经消失，下一帧在原地图继续定位和导航。
        view[1, 5] = 1
        resumed = maze.load_view(view, (1, 5))
        self.assertEqual(maze.char_coord, (1, 5))
        self.assertFalse(resumed.approaching_treasure)

    def test_no_dark_screen_preserves_completion_behavior(self):
        runner, capture, controller = self.make_runner()
        capture.grab.return_value = np.full((120, 600, 3), 100, dtype=np.uint8)
        with patch('zmxy4_treasure_hunt.maze.ft.time.sleep') as sleep:
            self.assertFalse(runner.next())
        self.assertEqual(capture.grab.call_count, 8)
        self.assertEqual(sleep.call_args_list, [call(0.5)] * 8)
        controller.stop.assert_called_once()
        controller.click.assert_not_called()

    def test_complete_map_ends_monitor_when_no_treasures_remain(self):
        runner, _, controller = self.make_runner()
        result = NavigationResult(np.zeros((5, 7), dtype=np.uint8), [], '', False, True)
        with patch.object(runner, 'step', return_value=result) as step, patch.object(runner, 'render'), \
             patch.object(runner, 'next') as check, \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.waitKey', return_value=-1):
            runner.monitor()
        step.assert_called_once()
        check.assert_not_called()

    def test_missing_weekly_map_does_not_create_capture_or_send_inputs(self):
        with patch('zmxy4_treasure_hunt.maze.ft.load_weekday_map', side_effect=FileNotFoundError('map missing')), \
             patch('zmxy4_treasure_hunt.maze.ft.ScreenCapture') as capture, \
             patch('zmxy4_treasure_hunt.maze.ft.MovementController') as controller:
            with self.assertRaises(FileNotFoundError):
                FindTreasure()
        capture.assert_not_called()
        controller.assert_not_called()

    def test_fake_treasure_darkness_detection_uses_raw_capture(self):
        runner, capture, _ = self.make_runner()
        capture.grab.return_value = np.zeros((120, 600, 3), dtype=np.uint8)
        with patch.object(runner.mask_restorer, 'restore', side_effect=AssertionError('不得反算黑屏检测帧')), \
             patch('zmxy4_treasure_hunt.maze.ft.time.sleep'):
            self.assertTrue(runner.next())

    def test_alignment_failure_stops_before_export_and_preserves_original_error(self):
        runner, capture, controller = self.make_runner(debug=True)
        capture.grab.return_value = np.zeros((400, 600, 3), np.uint8)
        failure = MapAlignmentError('alignment failed')
        events = Mock()
        events.attach_mock(controller.stop, 'stop')
        with patch('zmxy4_treasure_hunt.maze.ft.locate_character', return_value=Character(50, 50, 10)), \
             patch('zmxy4_treasure_hunt.maze.ft.detect_walls'), \
             patch.object(runner.maze, 'analyze', side_effect=failure), \
             patch('zmxy4_treasure_hunt.maze.ft.export_alignment_debug', return_value=('debug-path', {})) as export:
            events.attach_mock(export, 'export')
            with self.assertRaises(MapAlignmentError) as caught:
                runner.step()
        self.assertIs(caught.exception, failure)
        self.assertEqual(events.mock_calls[0], call.stop())
        export.assert_called_once()
        controller.apply.assert_not_called()
        self.assertTrue(any('debug-path' in note for note in failure.__notes__))

    def test_debug_export_failure_does_not_hide_original_error(self):
        runner, capture, controller = self.make_runner(debug=True)
        capture.grab.return_value = np.zeros((400, 600, 3), np.uint8)
        failure = DetectionError('missing character')
        with patch('zmxy4_treasure_hunt.maze.ft.locate_character', side_effect=failure), \
             patch('zmxy4_treasure_hunt.maze.ft.export_alignment_debug', side_effect=OSError('read only')):
            with self.assertRaises(DetectionError) as caught:
                runner.step()
        self.assertIs(caught.exception, failure)
        self.assertTrue(any('read only' in note for note in failure.__notes__))
        controller.stop.assert_called_once()

    def test_normal_mode_main_never_uses_debug_window(self):
        runner, capture, controller = self.make_runner()
        self.assertFalse(runner.debug)
        with patch.object(runner, 'monitor'), \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.namedWindow') as named, \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.moveWindow') as move, \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.resizeWindow') as resize, \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.setWindowProperty') as properties, \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.destroyAllWindows') as destroy, \
             patch('zmxy4_treasure_hunt.maze.ft.time.sleep'):
            runner.main()
        for operation in (named, move, resize, properties, destroy):
            operation.assert_not_called()
        controller.stop.assert_called_once()
        capture.close.assert_called_once()

    def test_normal_mode_monitor_never_renders_or_polls_gui_events(self):
        runner, _, _ = self.make_runner()
        result = NavigationResult(np.zeros((5, 7), np.uint8), [], '', False, True)
        with patch.object(runner, 'step', return_value=result), patch.object(runner, 'render') as render, \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.waitKey') as wait:
            runner.monitor()
        render.assert_not_called()
        wait.assert_not_called()

    def test_normal_mode_alignment_failure_never_exports_diagnostics(self):
        runner, capture, controller = self.make_runner()
        capture.grab.return_value = np.zeros((400, 600, 3), np.uint8)
        failure = DetectionError('missing character')
        with patch('zmxy4_treasure_hunt.maze.ft.locate_character', side_effect=failure), \
             patch('zmxy4_treasure_hunt.maze.ft.export_alignment_debug') as export:
            with self.assertRaises(DetectionError) as caught:
                runner.step()
        export.assert_not_called()
        self.assertIs(caught.exception, failure)
        self.assertFalse(hasattr(failure, '__notes__'))
        controller.stop.assert_called_once()

    def test_debug_monitor_renders_and_accepts_escape(self):
        runner, _, _ = self.make_runner(debug=True)
        result = NavigationResult(np.zeros((5, 7), np.uint8), [], '', False)
        with patch.object(runner, 'step', return_value=result) as step, \
             patch.object(runner, 'render') as render, \
             patch('zmxy4_treasure_hunt.maze.ft.cv2.waitKey', return_value=27):
            runner.monitor()
        step.assert_called_once()
        render.assert_called_once()

    def test_cli_debug_is_opt_in(self):
        from zmxy4_treasure_hunt.maze.cli import main

        for arguments, expected in (([], False), (['--debug'], True)):
            with self.subTest(arguments=arguments), patch('zmxy4_treasure_hunt.maze.ft.FindTreasure') as runner:
                self.assertEqual(main(arguments), 0)
                runner.assert_called_once_with(interval=0.05, debug=expected)
                runner.return_value.main.assert_called_once()
