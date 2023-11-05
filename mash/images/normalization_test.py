import unittest

import numpy as np

from mash.images import normalization


class TestImageStandardization(unittest.TestCase):
    def setUp(self):
        # Creating example images
        self.rgb_image = np.random.rand(224, 224, 3).astype(np.float32)
        self.single_channel_image = np.random.rand(224, 224, 1).astype(np.float32)

    def test_rgb_input_returns_normalized_image(self):
        mean = np.mean(self.rgb_image, axis=(0, 1))
        std = np.std(self.rgb_image, axis=(0, 1))
        normalized_image = normalization.standardize(self.rgb_image, mean=mean, std=std)

        # Check dtype and shape
        self.assertIsInstance(normalized_image[0, 0, 0], np.floating)
        self.assertEqual(normalized_image.shape, self.rgb_image.shape)

        # Check if the image is actually normalized
        self.assertTrue(
            np.allclose(np.mean(normalized_image, axis=(0, 1)), 0, atol=1e-2)
        )
        self.assertTrue(
            np.allclose(np.std(normalized_image, axis=(0, 1)), 1, atol=1e-2)
        )

    def test_unknown_dataset_raises_value_error(self):
        with self.assertRaises(ValueError) as context:
            normalization.standardize(
                np.random.rand(224, 224, 3), dataset="non_existent"
            )
        self.assertIn("Unknown dataset: non_existent", str(context.exception))

    def test_too_many_channels_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalization.standardize(
                np.concatenate([self.rgb_image, self.rgb_image[..., :1]], axis=-1),
                dataset="imagenet",
            )

    def test_both_dataset_and_mean_std_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalization.standardize(
                self.rgb_image,
                dataset="imagenet",
                mean=np.array([0.5]),
                std=np.array([0.5]),
            )

    def test_unequal_mean_std_and_channels_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalization.standardize(
                self.single_channel_image,
                mean=np.array([0.5, 0.5]),
                std=np.array([0.5, 0.5]),
            )

    def test_existing_dataset_returns_floating_type(self):
        image = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        normalized_image = normalization.standardize(image, dataset="imagenet")

        self.assertIsInstance(normalized_image[0, 0, 0], np.floating)

    def test_no_dataset_or_mean_std_raises_value_error(self):
        with self.assertRaises(ValueError):
            normalization.standardize(self.rgb_image)


if __name__ == "__main__":
    unittest.main()
