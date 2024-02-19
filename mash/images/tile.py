import numpy as np


def image_to_tiles(image: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    """Splits an image into tiles.

    Args:
        image: A numpy array representing the image.
        tile_width: The width of each tile.
        tile_height: The height of each tile.

    Returns:
        A numpy array containing the tiles. Each tile is a sub-array of the original image.
    """
    # Check for invalid inputs.
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Tile width and height must be positive integers")
    if not isinstance(tile_width, int) or not isinstance(tile_height, int):
        raise TypeError("Tile width and height must be integers")

    # Calculate the number of tiles in each dimension.
    num_tiles_x = image.shape[1] // tile_width
    num_tiles_y = image.shape[0] // tile_height

    # Extract the tiles.
    tiles = []
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            tile = image[
                y * tile_height : (y + 1) * tile_height,
                x * tile_width : (x + 1) * tile_width,
            ]
            tiles.append(tile)

    tiles_array = np.array(tiles)

    return tiles_array
