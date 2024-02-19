import unittest

import numpy as np

from mash.images import truecolor


class TransparentToRgbTests(unittest.TestCase):
    def test_3d_input_with_4_channels(self):
        image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        result = truecolor.transparent_to_rgb(image)
        self.assertEqual(result.shape, (100, 100, 3))
        np.testing.assert_array_equal(result, image[:, :, :3])

    def test_2d_input_raises(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            truecolor.transparent_to_rgb(image)

    def test_3d_input_with_3_channels_raises(self):
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            truecolor.transparent_to_rgb(image)


class GrayscaleToRgbTests(unittest.TestCase):
    def test_2d_input_duplicates_channels(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = truecolor.grayscale_to_rgb(image)
        self.assertEqual(result.shape, (100, 100, 3))
        for channel in range(3):
            np.testing.assert_array_equal(result[:, :, channel], image)

    def test_3d_input_duplicates_channels(self):
        image = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        result = truecolor.grayscale_to_rgb(image)
        self.assertEqual(result.shape, (100, 100, 3))
        for channel in range(3):
            np.testing.assert_array_equal(result[:, :, channel], image[:, :, 0])

    def test_4d_input_raises(self):
        image = np.random.randint(0, 256, (100, 100, 1, 1), dtype=np.uint8)
        with self.assertRaises(ValueError):
            truecolor.grayscale_to_rgb(image)


if __name__ == "__main__":
    unittest.main()
