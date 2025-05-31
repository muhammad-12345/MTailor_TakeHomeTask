# test.py

import unittest
import numpy as np
from PIL import Image
from model import Preprocessor, OnnxModel, ImageLoader


class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        self.model = OnnxModel("assets/model.onnx")
        self.preprocessor = Preprocessor()
        self.image1 = ImageLoader.load("assets/n01440764_tench.jpeg")
        self.image2 = ImageLoader.load("assets/n01667114_mud_turtle.jpeg")

    def test_image_loading(self):
        self.assertIsInstance(self.image1, Image.Image)
        self.assertIsInstance(self.image2, Image.Image)

    def test_preprocessing_shape(self):
        processed = self.preprocessor.process(self.image1)
        self.assertEqual(processed.shape, (1, 3, 224, 224))

    def test_inference_output(self):
        processed = self.preprocessor.process(self.image1)
        pred_class = self.model.predict(processed)
        self.assertIsInstance(pred_class, int)
        self.assertTrue(0 <= pred_class < 1000)

    def test_known_class_ids(self):
        pred1 = self.model.predict(self.preprocessor.process(self.image1))
        pred2 = self.model.predict(self.preprocessor.process(self.image2))
        print(f"Image1 class: {pred1}, Image2 class: {pred2}")
        self.assertIn(pred1, range(1000))
        self.assertIn(pred2, range(1000))


if __name__ == "__main__":
    unittest.main()
