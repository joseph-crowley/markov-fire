from typing import List, Tuple
from enum import Enum

import numpy as np
import random

class GridState(Enum):
    EMPTY = 0
    TREE = 1
    ON_FIRE = 2
    BURNED = 3
    PREVIOUSLY_BURNED = 4

class Environment:
    def __init__(self, perturbations: List[Tuple[Tuple[int, int], float]]):
        self.perturbations = perturbations

    def get_perturbation(self, grid: np.ndarray=None) -> Tuple[int, int]:
        if grid is not None:
            tree_mask = (grid == GridState.TREE.value)
            on_fire_mask = (grid == GridState.ON_FIRE.value)
            tree_coords = np.array(np.where(tree_mask)).T
            fire_coords = np.array(np.where(on_fire_mask)).T
            dist = np.zeros(grid.shape)
            for tree in tree_coords:
                distances = np.sum((fire_coords - tree) ** 2, axis=1)
                dist[tree[0], tree[1]] = np.sum(np.exp(-distances / (2**2)))
            dist = dist / np.sum(dist)
            dist = np.nan_to_num(dist, nan=1)
            perturbations = [((i,j), dist[i,j]) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
            self.perturbations = perturbations
        return random.choices(
            population=[p[0] for p in self.perturbations],
            weights=[p[1] for p in self.perturbations],
            k=1
        )[0]
