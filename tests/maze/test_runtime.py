import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from zmxy4_treasure_hunt.maze.capture import ScreenCapture
from zmxy4_treasure_hunt.maze.controls import MovementController
from zmxy4_treasure_hunt.maze.navigation import NavigationResult
from zmxy4_treasure_hunt.maze.ft import FindTreasure
from zmxy4_treasure_hunt.maze.vision import Character, DetectionError, WallObservation


class CaptureTests(unittest.TestCase):
    def test_retry_and_region_preserve_bgr_frame(self):
        camera = Mock()
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[..., 0] = 200
        camera.grab.side_effect = [None, frame]
        capture = ScreenCapture(camera)
        with patch('zmxy4_treasure_hunt.maze.capture.time.sleep'):
            self.assertIs(capture.grab((10, 20, 30, 30)), frame)
        camera.grab.assert_called_with(region=(10, 20, 30, 30), new_frame_only=False)

    def test_timeout_and_close_are_bounded_and_idempotent(self):
        camera = Mock()
        camera.grab.return_value = None
        capture = ScreenCapture(camera)
        with patch('zmxy4_treasure_hunt.maze.capture.time.monotonic', side_effect=[0, 5]):
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
        with patch('zmxy4_treasure_hunt.maze.controls.key_down') as down, patch('zmxy4_treasure_hunt.maze.controls.key_up') as up:
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
        with patch('zmxy4_treasure_hunt.maze.controls.key_up', side_effect=[OSError('input failed'), None, None, None]) as up:
            with self.assertRaises(OSError):
                controller.stop()
        self.assertEqual(up.call_args_list, [call(char) for char in 'WASD'])

    def test_invalid_movements_send_no_events(self):
        controller = MovementController()
        with patch('zmxy4_treasure_hunt.maze.controls.key_down') as down:
            with self.assertRaises(ValueError):
                controller.apply('Q')
        down.assert_not_called()


class RunnerTests(unittest.TestCase):
    def make_runner(self):
        capture, controller = Mock(), Mock()
        runner = FindTreasure(capture=capture, controller=controller, region=(10, 20, 610, 420))
        return runner, capture, controller

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
        with patch('zmxy4_treasure_hunt.maze.ft.locate_character', return_value=Character(50, 50, 10)), \
             patch('zmxy4_treasure_hunt.maze.ft.detect_walls', return_value=observation):
            result = runner.step()
        self.assertEqual(runner.maze.char_coord, (1, 1))
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
        runner, capture, controller = self.make_runner()
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
        runner, _, _ = self.make_runner()
        maze = runner.maze
        view = np.full((5, 7), 2, dtype=np.uint8)
        view[1, 1:6] = [1, 3, 1, 3, 4]
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
        # 恢复画面后假宝箱已经消失，下一帧在原地图继续定位和探索。
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
