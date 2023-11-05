import random

import numpy as np

from mash.images import truecolor


def crop_rectangle(
    image: np.ndarray,
    crop_height: int,
    crop_width: int,
    start_x: int,
    start_y: int,
    return_rgb: bool = False,
) -> np.ndarray:
    """Crop a rectangle from the image.

    Args:
        image: The image to crop, either grayscale or RGB/A.
        crop_height: The height of the crop.
        crop_width: The width of the crop.
        start_x: The x-coordinate of the top-left corner of the crop.
        start_y: The y-coordinate of the top-left corner of the crop.
        return_rgb: Return in RGB format instead of the input format.

    Returns:
        The cropped image.
    """
    # Calculate the expected output shape.
    expected_shape = list(image.shape)
    expected_shape[0] = crop_height
    expected_shape[1] = crop_width

    # We'll operate on the 3D version of the image, so expand if necessary.
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)

    if crop_height <= 0 or crop_width <= 0:
        raise ValueError("Crop size must be a positive integer.")
    if image.ndim != 3:
        raise ValueError("Unsupported image type, requires 2D or 3D array.")

    height, width, _ = image.shape
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Crop size should be smaller or equal to both image dimensions."
        )

    # Calculate the ending position of the crop.
    end_y = start_y + crop_height
    end_x = start_x + crop_width

    # Crop the image array using NumPy indexing.
    cropped_image = image[start_y:end_y, start_x:end_x, :]

    # If requested, reshape back to the original format.
    if return_rgb:
        if image.shape[2] == 3:
            return cropped_image
        elif image.shape[2] == 1:
            return truecolor.grayscale_to_rgb(cropped_image)
        else:
            return cropped_image[:, :, :3]
    else:
        return np.reshape(cropped_image, expected_shape)


def crop_square(
    image: np.ndarray,
    crop_size: int,
    start_x: int,
    start_y: int,
    return_rgb: bool = False,
) -> np.ndarray:
    """Crop a square from the image.

    Args:
        image: The image to crop.
        crop_size: The side length of the crop.
        start_x: The x-coordinate of the top-left corner of the crop.
        start_y: The y-coordinate of the top-left corner of the crop.
        return_rgb: Return in RGB format instead of the input format.

    Returns:
        The cropped image.
    """
    return crop_rectangle(
        image, crop_size, crop_size, start_x, start_y, return_rgb=return_rgb
    )


def center_square_crop(
    image: np.ndarray, crop_size: int, return_rgb: bool = False
) -> np.ndarray:
    """Crop the center of the image to the specified size.

    Args:
        image: The image to crop.
        crop_size: The size of the crop.
        return_rgb: Return in RGB format instead of the input format.

    Returns:
        The cropped image.
    """
    height, width = image.shape[:2]

    if crop_size > height or crop_size > width:
        raise ValueError("Side length is larger than image dimensions.")
    if crop_size <= 0:
        raise ValueError("Side length must be a positive integer.")

    # Compute crop based on rounded center of the image.
    y_start = (height - crop_size) // 2
    x_start = (width - crop_size) // 2

    return crop_square(image, crop_size, x_start, y_start, return_rgb=return_rgb)


def random_square_crop(
    image: np.ndarray, crop_size: int, return_rgb: bool = False
) -> np.ndarray:
    """Crop a random square from the image.

    Args:
        image: The image to crop.
        crop_size: The side length of the crop.
        return_rgb: Return in RGB format instead of the input format.

    Returns:
        The cropped image.
    """
    height, width = image.shape[:2]

    if crop_size > height or crop_size > width:
        raise ValueError("Side length is larger than image dimensions.")
    if crop_size <= 0:
        raise ValueError("Side length must be a positive integer.")

    # Calculate the maximum starting position for the top-left corner of the crop.
    max_y = height - crop_size
    max_x = width - crop_size

    # Randomly select the starting position of the crop.
    start_y = random.randint(0, max_y)
    start_x = random.randint(0, max_x)

    return crop_square(image, crop_size, start_x, start_y, return_rgb=return_rgb)


def crop_to_multiple_of_dimension(img: np.ndarray, multiple: int) -> np.ndarray:
    """Crop an image to a multiple of a dimension. Useful for models that require
    input dimensions that are multiples of a certain number, i.e. ViT models.

    Args:
        img: The image to crop.
        multiple: The multiple of each dimension to crop to.

    Returns:
        The cropped image, centered in the original image.
    """
    if multiple <= 0:
        raise ValueError("Multiple must be a positive integer.")

    if len(img.shape) != 3:
        raise ValueError("Unsupported image type, requires 3D array.")

    height, width, _ = img.shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive integers.")

    # Compute the new dimensions
    new_width = width - (width % multiple)
    new_height = height - (height % multiple)

    # Compute coordinates for the new image
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return img[top:bottom, left:right]
