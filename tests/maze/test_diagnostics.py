import json
from pathlib import Path
import unittest

import numpy as np

from zmxy4_treasure_hunt.maze.diagnostics import export_alignment_debug, rank_weekdays
from zmxy4_treasure_hunt.maze.map_templates import MapTemplate, load_weekday_map
from zmxy4_treasure_hunt.maze.mask import RestoredFrame
from zmxy4_treasure_hunt.maze.navigation import MapAlignmentError, Maze


class DiagnosticTests(unittest.TestCase):
    def corridor(self):
        cells = np.full((5, 7), 2, np.uint8)
        cells[1, 1:6] = [1, 3, 1, 3, 4]
        return cells

    def test_rejections_distinguish_bounds_parity_and_low_score(self):
        cells = self.corridor()
        cases = [('out_of_bounds', (1, 3)), ('grid_parity', (1, 1)), ('low_score', (1, 1))]
        for reason, player in cases:
            with self.subTest(reason=reason):
                maze = Maze(100, 100, cells)
                view = cells.copy()
                if reason == 'grid_parity':
                    maze.grid.vertical_start_small = False
                if reason == 'low_score':
                    view[1, 2] = view[1, 4] = 2
                    view[2, 1] = 3
                with self.assertRaises(MapAlignmentError) as caught:
                    maze.load_view(view, player)
                candidate = caught.exception.details['candidates'][0]
                self.assertEqual(candidate['reason'], reason)
                if reason == 'low_score':
                    self.assertEqual(len(candidate['mismatch_cells']), 3)
                    self.assertGreater(candidate['compared_cells'], 0)
                self.assertEqual(maze.char_coord, (1, 1))
                np.testing.assert_array_equal(maze.maze, cells)

    def test_false_treasure_is_separate_from_wall_score(self):
        cells = self.corridor()
        maze = Maze(100, 100, cells)
        view = cells.copy()
        view[1, 3] = 4
        with self.assertRaises(MapAlignmentError) as caught:
            maze.load_view(view, (1, 1))
        candidate = caught.exception.details['candidates'][0]
        self.assertEqual(candidate['wall_score'], 1)
        self.assertEqual(candidate['score'], 0)
        self.assertEqual(candidate['unexpected_treasures'], [[1, 3]])

    def test_weekday_ranking_is_diagnostic_only(self):
        cells = load_weekday_map(2).cells
        view = cells[:15, :15].copy()
        before = view.copy()
        results = rank_weekdays(view, (1, 1), (0, 0))
        self.assertEqual(results[2]['fixed_start']['score'], 1)
        self.assertLess(results[3]['fixed_start']['score'], 1)
        self.assertEqual(len(results), 7)
        np.testing.assert_array_equal(view, before)

    def test_export_contains_images_grid_state_and_rejection_details(self):
        cells = self.corridor()
        maze = Maze(100, 100, cells)
        view = cells.copy()
        view[1, 2] = view[1, 4] = 2
        view[2, 1] = 3
        try:
            maze.load_view(view, (1, 1))
        except MapAlignmentError as error:
            failure = error
        raw = np.full((100, 100, 3), 100, np.uint8)
        frame = RestoredFrame(raw.copy(), np.ones((100, 100), bool))
        root = Path(__file__).resolve().parents[2]
        parent = root / 'runs/test-maze-diagnostics'
        parent.mkdir(parents=True, exist_ok=True)
        # 保留可检查的诊断产物，避免依赖临时目录的系统 ACL。
        output, _ = export_alignment_debug(maze, MapTemplate(1, 'test', cells), failure,
                                           raw=raw, frame=frame, output_dir=parent / 'export-case')
        report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['observed_grid'], view.tolist())
        self.assertEqual(report['alignment']['candidates'][0]['reason'], 'low_score')
        for name in ('raw.png', 'restored.png', 'grid_overlay.png', 'observed.png', 'comparison.png'):
            self.assertTrue((output / name).is_file(), name)
