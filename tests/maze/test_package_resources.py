"""资源通过 Traversable 读取，不要求普通磁盘目录或项目根目录。"""

from importlib.resources import files
from io import BytesIO
import unittest
from unittest.mock import patch
from zipfile import Path as ZipPath, ZipFile

from zmxy4_treasure_hunt.maze import maps
from zmxy4_treasure_hunt.maze.map_templates import load_weekday_map
from zmxy4_treasure_hunt.maze.mask import MaskRestorer


class PackageResourceTests(unittest.TestCase):
    def test_map_can_be_loaded_from_zip_without_a_filesystem_path(self):
        memory = BytesIO()
        with ZipFile(memory, 'w') as archive:
            for weekday in range(7):
                archive.writestr(f'maps/{weekday}.json', files(maps).joinpath(f'{weekday}.json').read_bytes())
        memory.seek(0)
        with ZipFile(memory) as archive:
            directory = ZipPath(archive, at='maps/')
            with patch('zmxy4_treasure_hunt.maze.map_templates.files', return_value=directory):
                for weekday in range(7):
                    self.assertEqual(load_weekday_map(weekday).weekday, weekday)

    def test_mask_can_be_decoded_from_zip_without_cv2_imread(self):
        resource = files('zmxy4_treasure_hunt.maze').joinpath('assets', 'edge_mask.png')
        memory = BytesIO()
        with ZipFile(memory, 'w') as archive:
            archive.writestr('assets/edge_mask.png', resource.read_bytes())
        memory.seek(0)
        with ZipFile(memory) as archive:
            with patch('zmxy4_treasure_hunt.maze.mask.files', return_value=ZipPath(archive)):
                restorer = MaskRestorer(620, 344)
                self.assertEqual(restorer.transmission.shape, (344, 620))
                self.assertGreater(restorer.valid_pixels.mean(), 0.7)
