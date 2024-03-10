import unittest

import numpy as np

from mash.images import image_to_tiles


class TestImageToTilesComplete(unittest.TestCase):
    def test_perfectly_divisible(self):
        image = np.arange(100 * 100).reshape((100, 100))
        tiles = image_to_tiles(image, 10, 10)
        self.assertEqual(tiles.shape[0], 100)
        self.assertTrue(np.array_equal(tiles[0], image[:10, :10]))

    def test_not_perfectly_divisible_with_overlap(self):
        image = np.arange(105 * 105).reshape((105, 105))
        tiles = image_to_tiles(image, 10, 10, 5, 5)  # Including overlap
        # The calculation for the expected number of tiles considering overlap
        expected_tiles_x = (105 - 5) // (10 - 5)
        expected_tiles_y = (105 - 5) // (10 - 5)
        expected_total_tiles = expected_tiles_x * expected_tiles_y
        self.assertEqual(tiles.shape[0], expected_total_tiles)

    def test_large_tile_size(self):
        image = np.arange(30 * 30).reshape((30, 30))
        tiles = image_to_tiles(image, 40, 40)
        self.assertEqual(tiles.shape[0], 0)

    def test_small_image(self):
        image = np.arange(5 * 5).reshape((5, 5))
        tiles = image_to_tiles(image, 2, 2)
        self.assertEqual(tiles.shape[0], 4)

    def test_zero_dimension_tiles(self):
        image = np.arange(10 * 10).reshape((10, 10))
        with self.assertRaises(ValueError):
            tiles = image_to_tiles(image, 0, 0)

    def test_negative_dimension_tiles(self):
        image = np.arange(10 * 10).reshape((10, 10))
        with self.assertRaises(ValueError):
            tiles = image_to_tiles(image, -10, -10)

    def test_one_dimension_larger(self):
        image = np.arange(30 * 30).reshape((30, 30))
        tiles = image_to_tiles(image, 40, 20)
        self.assertEqual(tiles.shape[0], 0)

    def test_single_tile_exact_match(self):
        image = np.arange(10 * 10).reshape((10, 10))
        tiles = image_to_tiles(image, 10, 10)
        self.assertEqual(tiles.shape[0], 1)
        self.assertTrue(np.array_equal(tiles[0], image))

    def test_non_integer_tile_dimensions(self):
        image = np.arange(10 * 10).reshape((10, 10))
        with self.assertRaises(TypeError):
            tiles = image_to_tiles(image, 5.5, 5.5)
