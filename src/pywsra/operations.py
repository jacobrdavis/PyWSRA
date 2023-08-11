"""

"""

__all__ = [
    "rotate_xy",
]

from typing import Tuple

import numpy as np


def rotate_xy(x: np.array, y: np.array, theta: np.array) -> Tuple:
    """
    Perform 2D coordinate rotation of points (x, y).

    Args:
        x (np.array): x coordinate to rotate
        y (np.array): y coordinate to rotate
        theta (np.array): rotation angles in [rad]

    Returns:
        Tuple[np.array, np.array]: rotated x and y coordinates
    """
    x_rot = x*np.cos(theta) - y*np.sin(theta)
    y_rot = x*np.sin(theta) + y*np.cos(theta)

    return x_rot, y_rot
