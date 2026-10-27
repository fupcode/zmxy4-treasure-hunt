import unittest
import json
from pathlib import Path

import numpy as np

from zmxy4_treasure_hunt.maze.map_templates import load_weekday_map
from zmxy4_treasure_hunt.maze.navigation import Graph, MapAlignmentError, Maze


class NavigationTests(unittest.TestCase):
    def corridor(self):
        cells = np.full((5, 7), 2, dtype=np.uint8)
        cells[1, 1:6] = [1, 3, 1, 3, 4]
        return cells

    def test_nearest_treasure_uses_shortest_known_path(self):
        self.assertEqual(Graph(self.corridor()).bfs((1, 1)), [(1, 1), (1, 3), (1, 5)])

    def test_unknown_cells_are_never_explored(self):
        cells = self.corridor()
        cells[1, 2] = 0
        self.assertEqual(Graph(cells).bfs((1, 1)), [])
        with self.assertRaises(ValueError):
            Maze(100, 100, cells)

    def test_start_in_passage_connects_to_adjacent_rooms(self):
        self.assertEqual(Graph(self.corridor()).bfs((1, 2)), [(1, 2), (1, 3), (1, 5)])

    def test_boundary_nodes_never_wrap_or_leave_map(self):
        cells = np.ones((4, 4), dtype=np.uint8)
        graph = Graph(cells)
        self.assertEqual(graph.bfs((3, 3)), [])
        self.assertTrue(all(graph._inside(target) for targets in graph.graph.values() for target in targets))
        self.assertEqual(graph.bfs((-1, 1)), [])

    def test_first_frame_uses_initialized_start(self):
        for weekday in range(7):
            with self.subTest(weekday=weekday):
                template = load_weekday_map(weekday)
                maze = Maze(100, 100, template.cells)
                self.assertEqual(maze.char_coord, (1, 1))
                view = template.cells[:15, :15].copy()
                result = maze.load_view(view, (1, 1))
                self.assertEqual(maze.view_start, (0, 0))
                self.assertEqual(maze.char_coord, (1, 1))
                self.assertTrue(result.path)
                self.assertEqual(maze.maze[result.path[-1]], 4)
                np.testing.assert_array_equal(maze.maze, template.cells)
                self.assertEqual(result.image[1, 1], 5)
                self.assertEqual(list(maze.move_history), [])

    def test_first_frame_matches_view_with_odd_origin(self):
        template = load_weekday_map(1)
        maze = Maze(100, 100, template.cells)
        maze.grid.vertical_start_small = False
        maze.grid.horizontal_start_small = False
        maze.load_view(template.cells[1:, 1:], (0, 0))
        self.assertEqual(maze.view_start, (1, 1))
        self.assertEqual(maze.char_coord, (1, 1))

    def test_visible_treasure_is_a_landmark_but_missing_treasure_is_not_removed(self):
        cells = self.corridor()
        maze = Maze(100, 100, cells)
        view = cells.copy()
        view[1, 5] = 1
        result = maze.load_view(view, (1, 1))
        self.assertEqual(result.path[-1], (1, 5))
        self.assertEqual(maze.maze[1, 5], 4)

    def test_camera_scroll_keeps_global_location_and_map(self):
        cells = load_weekday_map(1).cells
        maze = Maze(100, 100, cells)
        maze.load_view(cells[:15, :15], (1, 1))
        self.assertEqual(maze.view_start, (0, 0))
        maze.record_movements('D')
        # 摄像机向右滚动一个连接格，局部格网奇偶随之切换。
        maze.grid.vertical_start_small = False
        maze.load_view(cells[:15, 1:16], (1, 2))
        self.assertEqual(maze.view_start, (0, 1))
        self.assertEqual(maze.char_coord, (1, 3))
        np.testing.assert_array_equal(maze.maze, cells)

    def test_fixed_start_disambiguates_repeated_map_patterns(self):
        cells = np.full((5, 9), 2, dtype=np.uint8)
        cells[1, 1:8] = [1, 3, 1, 3, 1, 3, 4]
        maze = Maze(100, 100, cells)
        maze.load_view(cells[:3, :3], (1, 1))
        self.assertEqual(maze.view_start, (0, 0))
        self.assertEqual(maze.char_coord, (1, 1))

    def test_initial_view_must_be_consistent_with_fixed_start(self):
        cells = self.corridor()
        maze = Maze(100, 100, cells)
        with self.assertRaises(MapAlignmentError):
            maze.load_view(cells, (1, 3))
        self.assertEqual(maze.char_coord, (1, 1))
        self.assertIsNone(maze.view_start)

    def test_bad_frame_does_not_corrupt_complete_map_or_location(self):
        cells = self.corridor()
        maze = Maze(100, 100, cells)
        maze.load_view(cells, (1, 1))
        view = cells.copy()
        view[1, 2] = view[1, 4] = 2
        view[2, 1] = 3
        with self.assertRaises(MapAlignmentError):
            maze.load_view(view, (1, 1))
        np.testing.assert_array_equal(maze.maze, cells)
        self.assertEqual(maze.char_coord, (1, 1))

    def test_consumed_treasure_never_reappears_even_if_seen_in_stale_frame(self):
        cells = self.corridor()
        maze = Maze(100, 100, cells)
        maze.load_view(cells, (1, 1))
        maze.load_view(cells, (1, 3))
        maze.remove_treasure((1, 5))
        result = maze.load_view(cells, (1, 5))
        self.assertEqual(maze.maze[1, 5], 1)
        self.assertEqual(cells[1, 5], 4)
        self.assertFalse(result.approaching_treasure)
        self.assertTrue(result.complete)
        self.assertEqual(result.movements, '')

    def test_direction_correction_and_no_path(self):
        maze = Maze(100, 100, self.corridor())
        maze.char_offset = (0.8, -0.8)
        self.assertEqual(maze.generate_movements([(1, 1), (1, 3)]), 'SD')
        self.assertEqual(maze.generate_movements([(1, 1), (3, 1)]), 'AS')
        self.assertEqual(maze.generate_movements([]), '')

    def test_already_at_treasure_triggers_check(self):
        maze = Maze(100, 100, self.corridor())
        maze.load_view(self.corridor(), (1, 1))
        result = maze.load_view(self.corridor(), (1, 5))
        self.assertTrue(result.approaching_treasure)
        self.assertEqual(result.movements, '')

    def test_all_weekly_maps_can_visit_all_treasures_without_exploration(self):
        for weekday in range(7):
            with self.subTest(weekday=weekday):
                cells = load_weekday_map(weekday).cells
                maze = Maze(100, 100, cells)
                player = (1, 1)
                visited = set()
                for _ in range(8):
                    result = maze.load_view(cells, player)
                    self.assertFalse(result.complete)
                    target = result.path[-1]
                    self.assertNotIn(target, visited)
                    for a, b in zip(result.path, result.path[1:]):
                        middle = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                        self.assertEqual(maze.maze[middle], 3)
                        self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 2)
                    visited.add(target)
                    player = target
                    maze.remove_treasure(target)
                done = maze.load_view(cells, player)
                self.assertTrue(done.complete)
                self.assertEqual(done.path, [])
                self.assertEqual(len(visited), 8)
                self.assertEqual(np.count_nonzero(cells == 4), 8)

    def test_recorded_vertical_scroll_is_not_blocked_by_horizontal_inputs(self):
        path = Path(__file__).with_name('fixtures') / 'camera_scroll_alignment.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        for history in (data['move_history'], []):
            with self.subTest(history=history):
                cells = load_weekday_map(data['weekday']).cells
                maze = Maze(991, 550, cells)
                maze.view_start = tuple(data['previous_view_start'])
                maze.char_coord = tuple(data['previous_player'])
                maze.move_history.extend(history)
                maze.grid.vertical_start_small = data['vertical_start_small']
                maze.grid.horizontal_start_small = data['horizontal_start_small']
                result = maze.load_view(np.array(data['observed_grid'], np.uint8), data['view_char'])
                self.assertEqual(maze.view_start, tuple(data['expected_view_start']))
                self.assertEqual(maze.char_coord, tuple(data['expected_player']))
                accepted = [item for item in maze.last_alignment['candidates'] if item['reason'] == 'accepted']
                self.assertEqual(len(accepted), 1)
                self.assertGreater(accepted[0]['score'], 0.95)
                self.assertEqual(maze.maze[result.path[-1]], 4)
                np.testing.assert_array_equal(maze.maze, cells)

    def test_horizontal_view_change_is_allowed_while_recent_inputs_are_vertical(self):
        cells = load_weekday_map(3).cells
        maze = Maze(100, 100, cells)
        maze.load_view(cells[:15, :15], (1, 1))
        for _ in range(3):
            maze.record_movements('W')
        maze.grid.vertical_start_small = False
        maze.load_view(cells[:15, 1:16], (1, 0))
        self.assertEqual(maze.view_start, (0, 1))
        self.assertEqual(maze.char_coord, (1, 1))
