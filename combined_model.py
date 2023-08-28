from temporal_model import WildfireSpreadProcess
from spatial_model import GridState, Environment
import numpy as np

class CombinedModel:
    def __init__(self, temporal_model: WildfireSpreadProcess, grid: np.ndarray, environment: Environment):
        self.temporal_model = temporal_model
        self.grid = grid
        self.environment = environment

    def run_simulation(self, initial_population: int, time_steps: int):
        population, footprint, extinguishment_time = self.temporal_model.simulate(initial_population, time_steps)

        grids = [self.grid]
        for t in range(1, len(population)):
            new_fire_cells = population[t] - population[t-1]
            new_grid = self.update_grid(grids[-1], new_fire_cells)
            grids.append(new_grid)

        return grids

    def update_grid(self, grid: np.ndarray, new_fire_cells: int):
        new_grid = grid.copy()
        for _ in range(new_fire_cells):
            perturbation = self.environment.get_perturbation(new_grid)
            new_grid[perturbation] = GridState.ON_FIRE.value
        return new_grid
