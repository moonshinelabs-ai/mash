import unittest

import numpy as np

from mash.images import resize


class TestImageResizing(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_resize_image_min_side_returns_correct_shape(self):
        resized_image = resize.resize_image_min_side(self.image, min_side_len=50)
        self.assertEqual(50, min(resized_image.shape[0:2]))

    def test_resize_image_max_side_returns_correct_shape(self):
        resized_image = resize.resize_image_max_side(self.image, max_side_len=150)
        self.assertEqual(150, max(resized_image.shape[0:2]))

    def test_negative_min_side_len_raises_value_error(self):
        with self.assertRaises(ValueError):
            resize.resize_image_min_side(self.image, min_side_len=-50)

    def test_negative_max_side_len_raises_value_error(self):
        with self.assertRaises(ValueError):
            resize.resize_image_max_side(self.image, max_side_len=-50)

    def test_preserve_range_true_keeps_original_range(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        resized_image = resize.resize_image_min_side(
            image, min_side_len=50, preserve_range=True
        )
        # Check if the values remain in the original range.
        self.assertTrue(np.all(resized_image == 255))

    def test_preserve_range_false_normalizes_values(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        resized_image = resize.resize_image_min_side(
            image, min_side_len=50, preserve_range=False
        )
        # Check if the values are normalized to [0, 1] range.
        self.assertTrue(np.all((resized_image <= 1) & (resized_image >= 0)))


if __name__ == "__main__":
    unittest.main()
