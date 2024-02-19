import os
import unittest

import numpy as np
import torch
from PIL import Image

from mash.images import conversion


class TestPilFromFile(unittest.TestCase):
    def setUp(self):
        self.pil_image = Image.new("RGB", (100, 100))
        self.test_file_path = "test_image.jpg"
        self.pil_image.save(self.test_file_path)

    def tearDown(self):
        os.remove(self.test_file_path)

    def test_path_returns_pil(self):
        img = conversion.pil_from_uri(self.test_file_path)
        self.assertIsInstance(img, Image.Image)


class TestToNumpy(unittest.TestCase):
    def setUp(self):
        self.npy_array = np.random.rand(100, 100, 3)
        self.tensor = torch.randn((100, 100, 3))
        self.pil_image = Image.new("RGB", (100, 100))
        self.test_file_path = "test_image.jpg"
        self.pil_image.save(self.test_file_path)

    def tearDown(self):
        os.remove(self.test_file_path)

    def test_path_returns_numpy(self):
        result = conversion.to_numpy(self.test_file_path)
        self.assertIsInstance(result, np.ndarray)

    def test_numpy_returns_numpy(self):
        result = conversion.to_numpy(self.npy_array)
        self.assertIs(result, self.npy_array)

    def test_pil_returns_numpy(self):
        result = conversion.to_numpy(self.pil_image)
        self.assertIsInstance(result, np.ndarray)

    def test_tensor_returns_numpy(self):
        result = conversion.to_numpy(self.tensor)
        self.assertIsInstance(result, np.ndarray)

    def test_invalid_input_type_raises_for_numpy_conversion(self):
        with self.assertRaises(TypeError):
            conversion.to_numpy(12345)


class TestToPIL(unittest.TestCase):
    def setUp(self):
        self.npy_array = np.random.rand(100, 100, 3)
        self.tensor = torch.randn((100, 100, 3))
        self.pil_image = Image.new("RGB", (100, 100))
        self.test_file_path = "test_image.jpg"
        self.pil_image.save(self.test_file_path)

    def tearDown(self):
        os.remove(self.test_file_path)

    def test_path_returns_pil(self):
        result = conversion.to_pil(self.test_file_path)
        self.assertIsInstance(result, Image.Image)

    def test_numpy_returns_pil(self):
        # Test float32
        result = conversion.to_pil(self.npy_array)
        self.assertIsInstance(result, Image.Image)

        # Test uint8
        uint8 = (self.npy_array * 255).astype(np.uint8)
        result = conversion.to_pil(uint8)
        self.assertIsInstance(result, Image.Image)

    def test_pil_returns_pil(self):
        result = conversion.to_pil(self.pil_image)
        self.assertIs(result, self.pil_image)

    def test_tensor_returns_pil(self):
        # Test float32
        result = conversion.to_pil(self.tensor)
        self.assertIsInstance(result, Image.Image)

        # Test uint8
        uint8 = (self.tensor * 255).byte()
        result = conversion.to_pil(uint8)
        self.assertIsInstance(result, Image.Image)

    def test_invalid_input_type_raises_for_pil_conversion(self):
        with self.assertRaises(TypeError):
            conversion.to_pil(12345)


class TestToTensor(unittest.TestCase):
    def setUp(self):
        self.npy_array = np.random.rand(100, 100, 3)
        self.tensor = torch.randn((100, 100, 3))
        self.pil_image = Image.new("RGB", (100, 100))
        self.test_file_path = "test_image.jpg"
        self.pil_image.save(self.test_file_path)

    def tearDown(self):
        os.remove(self.test_file_path)

    def test_path_returns_tensor(self):
        result = conversion.to_tensor(self.test_file_path)
        self.assertIsInstance(result, torch.Tensor)

    def test_numpy_returns_tensor(self):
        # Test float32
        result = conversion.to_tensor(self.npy_array)
        self.assertIsInstance(result, torch.Tensor)

        # Test uint8
        uint8 = (self.npy_array * 255).astype(np.uint8)
        result = conversion.to_tensor(uint8)
        self.assertIsInstance(result, torch.Tensor)

    def test_pil_returns_tensor(self):
        result = conversion.to_tensor(self.pil_image)
        self.assertIsInstance(result, torch.Tensor)

    def test_tensor_returns_tensor(self):
        # Test float32
        result = conversion.to_tensor(self.tensor)
        self.assertIs(result, self.tensor)

        # Test uint8
        uint8 = (self.tensor * 255).byte()
        result = conversion.to_tensor(uint8)
        self.assertIsInstance(result, torch.Tensor)

    def test_invalid_input_type_raises_for_tensor_conversion(self):
        with self.assertRaises(TypeError):
            conversion.to_tensor(12345)


if __name__ == "__main__":
    unittest.main()
