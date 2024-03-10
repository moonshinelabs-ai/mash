import numpy as np


def image_to_tiles(
    image: np.ndarray,
    tile_width: int,
    tile_height: int,
    overlap_width: int | None = None,
    overlap_height: int | None = None,
) -> np.ndarray:
    """Splits an image into tiles with optional overlap.

    Args:
        image: A numpy array representing the image.
        tile_width: The width of each tile.
        tile_height: The height of each tile.
        overlap_width: The overlap between tiles horizontally.
        overlap_height: The overlap between tiles vertically.

    Returns:
        A numpy array containing the tiles. Each tile is a sub-array of the original image.
    """
    # Check for invalid inputs.
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Tile width and height must be positive integers")
    if not isinstance(tile_width, int) or not isinstance(tile_height, int):
        raise TypeError("Tile width and height must be integers")
    if overlap_width is not None and (
        overlap_width < 0 or not isinstance(overlap_width, int)
    ):
        raise ValueError("Overlap width must be a non-negative integer")
    if overlap_height is not None and (
        overlap_height < 0 or not isinstance(overlap_height, int)
    ):
        raise ValueError("Overlap height must be a non-negative integer")

    # Adjust the stride for extracting tiles based on the overlap.
    stride_x = tile_width - overlap_width if overlap_width is not None else tile_width
    stride_y = (
        tile_height - overlap_height if overlap_height is not None else tile_height
    )

    # Ensure stride values are positive to avoid infinite loops.
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("Overlap must be less than the dimensions of the tile")

    # Calculate the number of tiles in each dimension.
    num_tiles_x = (
        (image.shape[1] - overlap_width) // stride_x
        if overlap_width is not None
        else image.shape[1] // tile_width
    )
    num_tiles_y = (
        (image.shape[0] - overlap_height) // stride_y
        if overlap_height is not None
        else image.shape[0] // tile_height
    )

    # Extract the tiles.
    tiles = []
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            tile = image[
                y * stride_y : y * stride_y + tile_height,
                x * stride_x : x * stride_x + tile_width,
            ]
            tiles.append(tile)

    tiles_array = np.array(tiles)

    return tiles_array
