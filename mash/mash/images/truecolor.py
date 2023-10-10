import numpy as np


def transparent_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a transparent image to RGB by dropping the transparency.

    Args:
        image: The transparent image.

    Returns:
        The RGB image.
    """
    if image.ndim != 3:
        raise ValueError("Input must be 3D.")
    if image.shape[2] != 4:
        raise ValueError("Input must have 4 channels.")

    return image[:, :, :3]


def grayscale_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to RGB by duplicating the channels.

    Args:
        image: The grayscale image.

    Returns:
        The RGB image.
    """
    if image.ndim not in (2, 3):
        raise ValueError("Input must be 2D or 3D.")

    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)

    return np.concatenate([image] * 3, axis=2)
