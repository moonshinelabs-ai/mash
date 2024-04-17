import numpy as np
from skimage.transform import resize


def _resize_image_fixed_side(
    image: np.ndarray, side_len: int, method: str = "max", preserve_range: bool = True
) -> np.ndarray:
    # Do some checking.
    if side_len <= 0:
        raise ValueError("Side length must be a positive integer.")

    # Need epsilon for floating point errors.
    epsilon = 1e-3
    height, width, _ = image.shape

    # Compute the scaling factor.
    if method == "max":
        scaling_factor = side_len / max(height, width)
    elif method == "min":
        scaling_factor = side_len / min(height, width)
    else:
        raise ValueError("Invalid method specified. Choose 'max' or 'min'.")

    # Compute the new dimensions plus an epsilon that will get truncated.
    new_height = int(height * scaling_factor + epsilon)
    new_width = int(width * scaling_factor + epsilon)

    assert new_height == side_len or new_width == side_len, "One side incorrect"

    resized_image = resize(
        image, (new_height, new_width), preserve_range=preserve_range
    )

    return resized_image


def resize_image_min_side(
    image: np.ndarray, min_side_len: int = 224, preserve_range: bool = True
) -> np.ndarray:
    """Resize the image such that the smallest side is equal to the specified length.

    Args:
        image: The image to resize.
        min_side_len: The length of the smallest side.
        preserve_range: Preserve the range of the image.

    Returns:
        The resized image.
    """
    return _resize_image_fixed_side(
        image, min_side_len, method="min", preserve_range=preserve_range
    )


def resize_image_max_side(
    image: np.ndarray, max_side_len: int = 224, preserve_range: bool = True
) -> np.ndarray:
    """Resize the image such that the longest side is equal to the specified length.

    Args:
        image: The image to resize.
        max_side_len: The length of the smallest side.
        preserve_range: Preserve the range of the image, False for 0-1.

    Returns:
        The resized image.
    """
    return _resize_image_fixed_side(
        image, max_side_len, method="max", preserve_range=preserve_range
    )
