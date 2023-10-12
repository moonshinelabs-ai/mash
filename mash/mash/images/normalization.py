import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

_DATASET_MEAN_STD = {
    "imagenet": {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
}


def standardize(
    image: np.ndarray,
    dataset: str | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
):
    """Standardize an image.

    Args:
        image: Image (numpy) to standardize.
        dataset: Dataset to use for standardization. Defaults to "imagenet".
        mean: Mean to use for standardization. Defaults to None, overrides dataset string.
        std: Standard deviation to use for standardization. Defaults to None, overrides dataset string.

    Returns:
        np.ndarray: Standardized image.
    """
    # Allow either a dataset string or a mean/std to be specified.
    if dataset and mean is not None and std is not None:
        raise ValueError("Cannot specify both dataset and mean/std")

    # If mean/std are specified, use those.
    if dataset:
        if dataset in _DATASET_MEAN_STD:
            mean = _DATASET_MEAN_STD[dataset]["mean"]
            std = _DATASET_MEAN_STD[dataset]["std"]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    if mean is None or std is None:
        raise ValueError("Must specify either dataset or mean/std")

    # Make sure the dataset mean/std are the right shape for the image.
    n_channels = image.shape[-1]
    if n_channels != len(mean) or n_channels != len(std):
        raise ValueError(
            f"Input image must have {len(mean)} channels, not {n_channels}."
        )

    # Convert to float range if it's bytes.
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Normalize the image
    normalized_image = (image - mean) / std

    return normalized_image
