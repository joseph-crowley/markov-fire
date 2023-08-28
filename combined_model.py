from temporal_model import WildfireSpreadProcess
from spatial_model import GridState, Environment, System, Optimization, Simulation
import numpy as np

class CombinedModel:
    def __init__(self, temporal_model: WildfireSpreadProcess, initial_grid: np.ndarray, environment: Environment, resources: list):
        self.temporal_model = temporal_model
        self.initial_grid = initial_grid
        self.environment = environment
        self.resources = resources
        self.system = System(self.initial_grid)
        self.optimization = Optimization(self.system, self.environment, self.resources)
        self.simulation = Simulation(self.system, self.environment, self.resources)

    def run_simulation(self, initial_population: int, time_steps: int):
        # Run the temporal simulation
        population, footprint, extinguishment_time = self.temporal_model.simulate(initial_population, time_steps)

        # Initialize the spatial simulation
        grids = [self.initial_grid]
        for t in range(1, len(population)):
            # Update the grid based on the change in population
            new_fire_cells = population[t] - population[t-1]
            new_grid = self.update_grid(grids[-1], new_fire_cells)
            grids.append(new_grid)

        return grids, extinguishment_time

    def update_grid(self, grid: np.ndarray, new_fire_cells: int):
        # Apply resources before updating the fire
        optimized_grid = self.optimization.apply_resources(grid, self.resources)
        
        # Spread the fire based on the new fire cells
        for _ in range(new_fire_cells):
            perturbation = self.environment.get_perturbation(optimized_grid)
            optimized_grid[perturbation] = GridState.ON_FIRE.value
        
        # Run the spatial simulation for one step
        new_grid = self.simulation.simulate_step(optimized_grid)
        
        return new_grid
