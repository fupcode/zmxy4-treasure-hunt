import unittest

import numpy as np

from zmxy4_treasure_hunt.maze.navigation import Graph, MapAlignmentError, Maze


class NavigationTests(unittest.TestCase):
    def corridor(self):
        cells = np.full((5, 7), 2, dtype=np.uint8)
        cells[1, 1:6] = [1, 3, 1, 3, 4]
        return cells

    def test_treasure_cell_can_be_reached(self):
        self.assertEqual(Graph(self.corridor()).bfs((1, 1)), [(1, 1), (1, 3), (1, 5)])

    def test_unknown_frontier_is_preferred_over_distant_treasure(self):
        cells = np.full((5, 11), 2, dtype=np.uint8)
        cells[1, 1:10] = [4, 3, 1, 3, 1, 3, 1, 0, 0]
        self.assertEqual(Graph(cells).bfs((1, 7)), [(1, 7), (1, 9)])

    def test_nearby_treasure_is_preferred_over_unknown_frontier(self):
        cells = self.corridor()
        cells[1, 1] = 0
        self.assertEqual(Graph(cells).bfs((1, 3)), [(1, 3), (1, 5)])

    def test_start_in_passage_connects_to_adjacent_cells(self):
        self.assertEqual(Graph(self.corridor()).bfs((1, 2)), [(1, 2), (1, 3), (1, 5)])

    def test_boundary_nodes_never_wrap_or_leave_map(self):
        cells = np.ones((4, 4), dtype=np.uint8)
        graph = Graph(cells)
        self.assertEqual(graph.bfs((3, 3)), [])
        self.assertTrue(all(graph._inside(target) for targets in graph.graph.values() for target in targets))
        self.assertEqual(graph.bfs((-1, 1)), [])

    def test_first_frame_uses_observed_player_and_does_not_modify_view(self):
        maze = Maze(100, 100)
        view = self.corridor()
        original = view.copy()
        result = maze.load_view(view, (1, 3))
        self.assertEqual(maze.char_coord, (1, 3))
        self.assertEqual(result.movements, 'D')
        self.assertTrue(result.approaching_treasure)
        self.assertEqual(result.image[1, 3], 5)
        self.assertEqual(maze.maze[1, 3], 1)
        np.testing.assert_array_equal(view, original)
        self.assertEqual(list(maze.move_history), [])

    def test_first_frame_aligns_road_parity(self):
        maze = Maze(100, 100)
        maze.grid.vertical_start_small = False
        maze.grid.horizontal_start_small = False
        view = np.array([[1, 3, 4], [2, 2, 2]], dtype=np.uint8)
        result = maze.load_view(view, (0, 0))
        self.assertEqual(maze.char_coord, (1, 1))
        self.assertEqual(result.movements, 'D')
        self.assertTrue(result.approaching_treasure)

    def test_moving_view_aligns_using_executed_history(self):
        maze = Maze(100, 100)
        maze.maze = np.full((29, 29), 2, dtype=np.uint8)
        view = np.array([[1, 3, 1], [2, 2, 2]], dtype=np.uint8)
        maze.maze[0:2, 2:5] = view
        maze.record_movements('D')
        self.assertEqual(maze.slide(view), (0, 2))

    def test_alignment_failure_preserves_known_map(self):
        maze = Maze(100, 100)
        maze.maze = np.full((29, 29), 2, dtype=np.uint8)
        original = maze.maze.copy()
        with self.assertRaises(MapAlignmentError):
            maze.load_view(np.ones((3, 3), dtype=np.uint8), (1, 1))
        np.testing.assert_array_equal(maze.maze, original)

    def test_direction_correction_and_no_path(self):
        maze = Maze(100, 100)
        maze.char_offset = (0.8, -0.8)
        self.assertEqual(maze.generate_movements([(1, 1), (1, 3)]), 'SD')
        self.assertEqual(maze.generate_movements([(1, 1), (3, 1)]), 'AS')
        self.assertEqual(maze.generate_movements([]), '')

    def test_already_at_treasure_triggers_transition(self):
        maze = Maze(100, 100)
        result = maze.load_view(self.corridor(), (1, 5))
        self.assertTrue(result.approaching_treasure)
        self.assertEqual(result.movements, '')
