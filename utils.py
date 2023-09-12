from scipy.ndimage import gaussian_filter
import numpy as np

from enum import Enum

class GridState(Enum):
    EMPTY = 0
    TREE = 1
    ON_FIRE = 2
    BURNED = 3
    PREVIOUSLY_BURNED = 4
    FIREBREAK = 5
    REMOVED = 6  # State for removed trees due to thinning
    SUPPRESSED = 7  # State for suppressed fires
    PREVIOUSLY_SUPPRESSED = 8  # State for previously suppressed fires

def calculate_fire_proximity(grid: np.ndarray, variance: float = 4.0) -> np.ndarray:
    """
    Calculate a grid representing the proximity to fire cells.

    Parameters:
    - grid (np.ndarray): Input grid with fire locations (GridState.ON_FIRE)
    - variance (float): Variance for the Gaussian filter (higher values mean smoother transitions)

    Returns:
    - np.ndarray: A grid with values between 0 and 1 representing proximity to fire
    """
    proximity_grid = np.zeros_like(grid, dtype=float)

    # Mark fire locations with 1.0
    fire_coords = np.argwhere(grid == GridState.ON_FIRE)
    for x, y in fire_coords:
        proximity_grid[x, y] = 1.0

    # Apply Gaussian filter to smooth out the values
    proximity_grid = gaussian_filter(proximity_grid, variance)

    # Set fire locations to 1.0 again (in case they were smoothed out) and push out the middle with a power function
    proximity_grid[grid == GridState.ON_FIRE] = 1.0
    proximity_grid = np.power(proximity_grid, 2)

    # Normalize the grid to have values between 0 and 1
    max_val = np.max(proximity_grid)
    if max_val > 0:  # Avoid division by zero
        proximity_grid /= max_val

    return proximity_grid

def get_perimeter_cells(grid: np.ndarray):
    """get the cells at the perimeter of the fire"""
    perimeter_cells = set()

    on_fire_cells = np.argwhere(grid == GridState.ON_FIRE)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                  (-1, -1), (-1, 1), (1, -1), (1, 1)] 

    for i, j in on_fire_cells:
        for dx, dy in directions:
            ni, nj = i + dx, j + dy
            if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                if grid[ni, nj] == GridState.TREE:
                    perimeter_cells.add((i, j))
                    break

    return perimeter_cells