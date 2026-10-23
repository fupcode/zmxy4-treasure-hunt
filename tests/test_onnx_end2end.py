"""无需模型权重，验证 800×512 Stretch、归一化坐标和免 NMS 解析。"""

import unittest
from unittest.mock import Mock

import numpy as np

from zaomeng_hanghai.core.onnx_model import OnnxModel


class EndToEndTests(unittest.TestCase):
    def make_model(self):
        model = OnnxModel.__new__(OnnxModel)
        model.input_shape = [1, 3, 512, 800]
        model.input_name = "images"
        model.class_names = {0: "bullet", 1: "player"}
        model.session = Mock()
        return model

    def test_stretch_and_rgb_without_padding(self):
        model = self.make_model()
        image = np.zeros((1000, 1600, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        tensor = model.preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 512, 800))
        np.testing.assert_allclose(tensor[0, 0], 1)
        np.testing.assert_allclose(tensor[0, 1:], 0)

    def test_coordinates_filtering_and_overlaps(self):
        model = self.make_model()
        model.session.run.return_value = [np.array([[
            [100, 64, 200, 192, 0.9, 0],
            [100, 64, 200, 192, 0.8, 0],
            [100, 64, 200, 192, 0.1, 1],
            [0, 0, 800, 512, 0.7, 1],
        ]], dtype=np.float32)]
        result = model(np.zeros((1000, 1600, 3), dtype=np.uint8), conf_threshold=0.3)
        self.assertEqual(len(result["bullet"]), 2)
        self.assertEqual(len(result["player"]), 1)
        np.testing.assert_allclose(result["bullet"][0].xyxyn, [0.125, 0.125, 0.25, 0.375])
        np.testing.assert_allclose(result["bullet"][0].xywhn, [0.1875, 0.25, 0.125, 0.25])
        np.testing.assert_allclose(result["player"][0].xyxyn, [0, 0, 1, 1])
        np.testing.assert_allclose(result["player"][0].xywhn, [0.5, 0.5, 1, 1])

    def test_empty_output(self):
        model = self.make_model()
        model.session.run.return_value = [np.zeros((1, 0, 6), dtype=np.float32)]
        result = model(np.zeros((512, 800, 3), dtype=np.uint8))
        self.assertEqual(result, {"bullet": [], "player": []})

    def test_reject_raw_candidate_output(self):
        model = self.make_model()
        model.session.run.return_value = [np.zeros((1, 6, 8400), dtype=np.float32)]
        with self.assertRaises(ValueError):
            model(np.zeros((512, 800, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
