from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


def _maybe_convert_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype not in (np.uint8, np.float16, np.float32, np.float64):
        raise ValueError(f"Image must be of type uint8 or float32, got {image.dtype}")

    if image.dtype in (np.float32, np.float16, np.float64):
        image = (image * 255).astype(np.uint8)

    return image


def pil_from_uri(uri: str) -> Image.Image:
    """Return a PIL image from an image file or URL.

    Args:
        uri: File path or url to the image.

    Returns:
        PIL image.
    """
    if uri.startswith("http://") or uri.startswith("https://"):
        # The uri is a URL
        response = requests.get(uri)
        response.raise_for_status()  # Raise an error for bad responses
        return Image.open(BytesIO(response.content))
    else:
        # The uri is a file path
        return Image.open(uri)


def to_numpy(image: str | np.ndarray | Image.Image | torch.Tensor) -> np.ndarray:
    """Create a numpy array from a variety of input types.

    Args:
        input: Input to convert to a numpy array, can be a file path or an array.

    Returns:
        Numpy array, either uint8 for PIL images or the same type as the input.
    """
    if isinstance(image, str):
        pil_image = pil_from_uri(image)
        return np.array(pil_image)
    elif isinstance(image, np.ndarray):
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, torch.Tensor):
        return image.cpu().numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(image)}")


def to_pil(image: str | np.ndarray | Image.Image | torch.Tensor) -> Image.Image:
    """Create a PIL Image from a variety of input types.

    Args:
        input: Input to convert to a numpy array, can be a file path or an array.

    Returns:
        PIL image, if the input is not uint8 it will be assumed to be 0-1 and scaled.
    """
    if isinstance(image, str):
        return pil_from_uri(image)
    elif isinstance(image, np.ndarray):
        image = _maybe_convert_to_uint8(image)
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image
    elif isinstance(image, torch.Tensor):
        numpy = image.cpu().numpy()
        numpy = _maybe_convert_to_uint8(numpy)
        return Image.fromarray(numpy)
    else:
        raise TypeError(f"Unsupported input type: {type(image)}")


def to_tensor(image: str | np.ndarray | Image.Image | torch.Tensor) -> torch.Tensor:
    """Create a torch.Tensor from a variety of input types.

    Args:
        input: Input to convert to a tensor, can be a file path or an array.

    Returns:
        PyTorch Tensor, either uint8 for PIL images or the same type as the input.
    """
    if isinstance(image, str):
        pil_image = pil_from_uri(image)
        np_array = np.array(pil_image)
        return torch.from_numpy(np_array)
    elif isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    elif isinstance(image, Image.Image):
        np_array = np.array(image)
        return torch.from_numpy(np_array)
    elif isinstance(image, torch.Tensor):
        return image
    else:
        raise TypeError(f"Unsupported input type: {type(image)}")
