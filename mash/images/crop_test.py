import random
import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
from parameterized import parameterized

from mash.images import crop


class TestCenterSquareCrop(unittest.TestCase):
    input_sizes = [(10, 10), (10, 10, 1), (10, 10, 3), (10, 10, 4), (10, 10, 8)]
    crop_sizes = [1, 3, 5]

    @parameterized.expand(product(input_sizes, crop_sizes))
    def test_center_square_crop_given_input_shapes(
        self,
        input_shape: tuple[int, ...],
        crop_size: int,
    ):
        image_np = np.zeros(input_shape)
        cropped_image = crop.center_square_crop(image_np, crop_size, return_rgb=False)

        expected_shape: tuple[int, ...] = (crop_size, crop_size)
        if len(input_shape) == 3:
            expected_shape = expected_shape + (input_shape[2],)

        self.assertEqual(cropped_image.shape, expected_shape)

    @parameterized.expand(product(input_sizes, crop_sizes))
    def test_center_square_crop_given_input_shapes_force_rgb(
        self,
        input_shape: tuple[int, ...],
        crop_size: int,
    ):
        image_np = np.zeros(input_shape)
        cropped_image = crop.center_square_crop(image_np, crop_size, return_rgb=True)

        expected_shape = (crop_size, crop_size, 3)
        self.assertEqual(cropped_image.shape, expected_shape)

    @parameterized.expand(
        [
            ["too_large", (10, 10, 3), 15],
            ["negative", (10, 10, 3), -10],
            ["extra_dim", (10, 10, 3, 1), 5],
        ]
    )
    def test_center_square_crop_invalid_input_raises(
        self, name: str, input_shape: tuple[int, ...], crop_size: int
    ):
        image_np = np.zeros(input_shape)
        with self.assertRaises(ValueError):
            crop.center_square_crop(image_np, crop_size)

    def test_center_square_crop_correct_values(self):
        image_np = np.reshape(np.arange(16), (4, 4))
        cropped_image = crop.center_square_crop(image_np, 2).tolist()
        expected_result = [[5, 6], [9, 10]]

        self.assertListEqual(cropped_image, expected_result)


class TestCropSquare(unittest.TestCase):
    input_sizes = [(10, 10), (10, 10, 1), (10, 10, 3), (10, 10, 4), (10, 10, 8)]
    crop_sizes = [1, 3, 5]
    start_xs = [0, 2, 4]
    start_ys = [0, 2, 4]

    @parameterized.expand(product(input_sizes, crop_sizes, start_xs, start_ys))
    def test_center_square_crop_given_input_shapes(
        self,
        input_shape: tuple[int, ...],
        crop_size: int,
        start_x: int,
        start_y: int,
    ):
        image_np = np.zeros(input_shape)
        cropped_image = crop.crop_square(image_np, crop_size, start_x, start_y)

        expected_shape: tuple[int, ...] = (crop_size, crop_size)
        if len(input_shape) == 3:
            expected_shape = expected_shape + (input_shape[2],)

        self.assertEqual(cropped_image.shape, expected_shape)

    @parameterized.expand(
        [
            ["too_large", (10, 10, 3), 15, 0, 0],
            ["negative", (10, 10, 3), -10, 0, 0],
            ["extra_dim", (10, 10, 3, 1), 5, 0, 0],
            ["out_of_bounds_x", (10, 10, 3), 5, 10, 0],
            ["out_of_bounds_y", (10, 10, 3), 5, 0, 10],
        ]
    )
    def test_crop_square_invalid_input_raises(
        self,
        name: str,
        input_shape: tuple[int, ...],
        crop_size: int,
        start_x: int,
        start_y: int,
    ):
        image_np = np.zeros(input_shape)
        with self.assertRaises(ValueError):
            crop.crop_square(image_np, crop_size, start_x, start_y)

    def test_crop_square_correct_values(self):
        image_np = np.reshape(np.arange(16), (4, 4))
        cropped_image = crop.crop_square(image_np, 2, 0, 0).tolist()
        expected_result = [[0, 1], [4, 5]]

        self.assertListEqual(cropped_image, expected_result)


class TestCropRectangle(unittest.TestCase):
    input_sizes = [(10, 10), (10, 10, 1), (10, 10, 3), (10, 10, 4), (10, 10, 8)]
    starts = [0, 2, 4]
    sizes = [2, 4, 6]

    @parameterized.expand(product(input_sizes, starts, sizes))
    def test_center_square_crop_given_input_shapes(
        self,
        input_shape: tuple[int, ...],
        start: int,
        crop_size: int,
    ):
        image_np = np.zeros(input_shape)
        cropped_image = crop.crop_rectangle(
            image_np,
            start_x=start,
            start_y=start,
            crop_width=crop_size,
            crop_height=crop_size,
        )

        expected_shape: tuple[int, ...] = (crop_size, crop_size)
        if len(input_shape) == 3:
            expected_shape = expected_shape + (input_shape[2],)

        self.assertEqual(cropped_image.shape, expected_shape)

    @parameterized.expand(
        [
            ["too_large", (10, 10, 3), 15, 0, 0],
            ["negative", (10, 10, 3), -10, 0, 0],
            ["extra_dim", (10, 10, 3, 1), 5, 0, 0],
            ["out_of_bounds_x", (10, 10, 3), 5, 10, 0],
            ["out_of_bounds_y", (10, 10, 3), 5, 0, 10],
        ]
    )
    def test_crop_square_invalid_input_raises(
        self,
        name: str,
        input_shape: tuple[int, ...],
        crop_size: int,
        start_x: int,
        start_y: int,
    ):
        image_np = np.zeros(input_shape)
        with self.assertRaises(ValueError):
            crop.crop_rectangle(
                image_np,
                crop_width=crop_size,
                crop_height=crop_size,
                start_x=start_x,
                start_y=start_y,
            )

    def test_crop_square_correct_values(self):
        image_np = np.reshape(np.arange(16), (4, 4))
        cropped_image = crop.crop_rectangle(image_np, 2, 3, 0, 0).tolist()
        expected_result = [[0, 1, 2], [4, 5, 6]]

        self.assertListEqual(cropped_image, expected_result)


class TestRandomSquareCrop(unittest.TestCase):
    def test_random_square_crop_correct_size(self):
        image = np.random.randint(0, 255, size=(10, 10, 3), dtype="uint8")
        with patch.object(random, "randint", return_value=0):
            cropped_image = crop.random_square_crop(image, 5)

        self.assertEqual(cropped_image.shape, (5, 5, 3))

    def test_random_square_crop_incorrect_size_raises(self):
        image = np.random.randint(0, 255, size=(10, 10, 3), dtype="uint8")
        with self.assertRaises(ValueError):
            crop.random_square_crop(image, 15)

    def test_random_square_crop_zero_size_raises(self):
        image = np.random.randint(0, 255, size=(10, 10, 3), dtype="uint8")
        with self.assertRaises(ValueError):
            crop.random_square_crop(image, 0)

    def test_random_square_crop_negative_size_raises(self):
        image = np.random.randint(0, 255, size=(10, 10, 3), dtype="uint8")
        with self.assertRaises(ValueError):
            crop.random_square_crop(image, -5)

    def test_random_square_crop_with_side_length_equal_to_height(self):
        image = np.random.randint(0, 255, size=(10, 10, 3), dtype="uint8")
        with patch.object(random, "randint", return_value=0):
            cropped_image = crop.random_square_crop(image, 10)

        self.assertEqual(cropped_image.shape, (10, 10, 3))

    def test_random_square_crop_with_side_length_equal_to_width(self):
        image = np.random.randint(0, 255, size=(10, 10, 3), dtype="uint8")

        with patch.object(random, "randint", return_value=0):
            cropped_image = crop.random_square_crop(image, 10)

        self.assertEqual(cropped_image.shape, (10, 10, 3))


class TestCropToMultipleOfDimension(unittest.TestCase):
    def test_crop_result_has_correct_dimensions(self):
        img = np.zeros((15, 25, 3), dtype=np.uint8)
        multiple = 4
        cropped_img = crop.crop_to_multiple_of_dimension(img, multiple)

        # Check the shape of the cropped image
        self.assertEqual(cropped_img.shape[0] % multiple, 0)
        self.assertEqual(cropped_img.shape[1] % multiple, 0)

    def test_non_multiple_input_raises(self):
        img = np.zeros((15, 25, 3), dtype=np.uint8)

        # Assert that ValueError is raised for non-multiple dimension
        with self.assertRaises(ValueError):
            crop.crop_to_multiple_of_dimension(img, -1)
        with self.assertRaises(ValueError):
            crop.crop_to_multiple_of_dimension(img, 0)

    def test_img_input_validity(self):
        # Prepare an invalid image (with one dimension being a non-positive integer)
        invalid_img = np.zeros((0, 10, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            crop.crop_to_multiple_of_dimension(invalid_img, 4)

        invalid_img_2d = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(ValueError):
            crop.crop_to_multiple_of_dimension(invalid_img_2d, 4)


if __name__ == "__main__":
    unittest.main()
