"""校验图片提取产物可被当前导航编码读取，且地图、宝箱与星期保持一致。"""

import json
from datetime import date
from importlib.resources import files
from unittest.mock import patch
import unittest

import numpy as np

from zmxy4_treasure_hunt.maze import maps
from zmxy4_treasure_hunt.maze.navigation import Graph
from zmxy4_treasure_hunt.maze.navigation.templates import current_weekday, load_weekday_map

MAPS = files(maps)


class MapTemplateTests(unittest.TestCase):
    def load(self, weekday):
        data = json.loads((MAPS / f"{weekday}.json").read_text(encoding="utf-8"))
        cells = np.array(
            [[data["legend"][cell] for cell in row] for row in data["grid"]],
            dtype=np.uint8,
        )
        return data, cells

    def test_weekday_shape_and_navigation_encoding(self):
        for weekday in range(7):
            with self.subTest(weekday=weekday):
                data, cells = self.load(weekday)
                self.assertEqual(data["weekday"], weekday)
                self.assertEqual(data["source_image"], f"images/maze/ft{weekday}.png")
                self.assertEqual(data["size"], [29, 29])
                self.assertEqual(cells.shape, (29, 29))
                self.assertNotIn("start", data)
                self.assertEqual(set(np.unique(cells)), {1, 2, 3, 4})
                self.assertTrue(np.all(cells[[0, -1], :] == 2))
                self.assertTrue(np.all(cells[:, [0, -1]] == 2))
                self.assertTrue(np.all(cells[::2, ::2] == 2))
                self.assertTrue(np.all(np.isin(cells[1::2, 1::2], [1, 4])))

    def test_eight_treasures_match_room_coordinates(self):
        for weekday in range(7):
            with self.subTest(weekday=weekday):
                data, cells = self.load(weekday)
                detected = np.argwhere(cells == 4).tolist()
                self.assertEqual(len(detected), 8)
                self.assertEqual(detected, data["treasures"])
                self.assertTrue(all(row % 2 == col % 2 == 1 for row, col in detected))

    def test_all_rooms_and_treasures_are_reachable_by_existing_graph(self):
        for weekday in range(7):
            with self.subTest(weekday=weekday):
                data, cells = self.load(weekday)
                graph = Graph(cells)
                seen = {(1, 1)}
                pending = [(1, 1)]
                while pending:
                    for target in graph.graph[pending.pop()]:
                        if target not in seen:
                            seen.add(target)
                            pending.append(target)
                self.assertEqual(len(seen), 196)
                self.assertTrue(all(tuple(cell) in seen for cell in data["treasures"]))
                path = graph.bfs((1, 1))
                self.assertEqual(cells[path[-1]], 4)

    def test_weekday_numbering_and_automatic_selection(self):
        self.assertEqual(current_weekday(date(2026, 11, 1)), 0)
        self.assertEqual(current_weekday(date(2026, 11, 2)), 1)
        self.assertEqual(current_weekday(date(2026, 10, 31)), 6)
        for weekday in range(7):
            with self.subTest(weekday=weekday), patch(
                    'zmxy4_treasure_hunt.maze.navigation.templates.current_weekday', return_value=weekday):
                template = load_weekday_map()
                self.assertEqual(template.weekday, weekday)
                self.assertFalse(template.cells.flags.writeable)

    def test_invalid_or_missing_map_is_reported_without_exploration_fallback(self):
        with patch('pathlib.Path.read_text', side_effect=FileNotFoundError('missing map')):
            with self.assertRaises(FileNotFoundError):
                load_weekday_map(1)
        data, _ = self.load(1)
        data['weekday'] = 2
        with patch('pathlib.Path.read_text', return_value=json.dumps(data)):
            with self.assertRaisesRegex(ValueError, '星期编号'):
                load_weekday_map(1)
        data['weekday'] = 1
        data['grid'][1] = data['grid'][1].replace('.', '?', 1)
        with patch('pathlib.Path.read_text', return_value=json.dumps(data)):
            with self.assertRaises((ValueError, KeyError)):
                load_weekday_map(1)
