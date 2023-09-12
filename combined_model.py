from temporal_dynamics import WildfireSpreadModel
from spatial_dynamics import Environment
from utils import GridState

import numpy as np

class CombinedModel:
    def __init__(self, wildfire_model: WildfireSpreadModel, environment: Environment):
        self.wildfire_model = wildfire_model
        self.environment = environment

        self.updated_cells = set()

    def _distribute_firelets(self, num_firelets: int):
        # Identify the perimeter cells from where the fire can spread
        perimeter_cells = set(self.environment.perimeter_cells) - self.updated_cells
        
        # Identify the tree cells that are not on fire
        tree_cells = set(tuple(cell) for cell in np.argwhere(self.environment.grid == GridState.TREE.value)) - self.updated_cells

        perimeter_cells.update(tree_cells)
        
        # Get the fire proximity grid
        fire_proximity = self.environment.get_grid('fire_proximity')
        
        # Sort perimeter cells based on fire proximity
        sorted_perimeter_cells = sorted(perimeter_cells, key=lambda x: fire_proximity[x], reverse=True)
        
        # Distribute firelets based on sorted proximity
        for cell in sorted_perimeter_cells[:num_firelets]:
            x, y = cell
            self.environment.grid[x, y] = GridState.ON_FIRE.value
            self.updated_cells.add(cell)

    def _distribute_extinguishments(self, num_extinguishments: int):
        # Identify cells currently on fire
        on_fire_cells = set(tuple(cell) for cell in np.argwhere((self.environment.grid == GridState.ON_FIRE.value) | (self.environment.grid == GridState.BURNED.value)))
        on_fire_cells -= self.updated_cells  # Assuming self.updated_cells is also a set of tuples
        
        # Get the fuel moisture grid
        fuel_moisture = self.environment.get_grid('fuel_moisture')
        
        # Sort on-fire cells based on fuel moisture (higher moisture -> more likely to extinguish)
        sorted_on_fire_cells = sorted(on_fire_cells, key=lambda x: fuel_moisture[tuple(x)], reverse=True)
        
        # Distribute extinguishments
        for cell in sorted_on_fire_cells[:num_extinguishments]:
            x, y = cell
            self.environment.grid[x, y] = GridState.PREVIOUSLY_BURNED.value
            self.updated_cells.add(cell)

    def _distribute_suppressions(self, num_suppressions: int):
        # Identify cells currently on fire
        on_fire_cells = set(tuple(cell) for cell in np.argwhere((self.environment.grid == GridState.ON_FIRE.value) | (self.environment.grid == GridState.BURNED.value)))
        on_fire_cells -= self.updated_cells  # Assuming self.updated_cells is also a set of tuples

        # Get the cost and potency grids
        cost = self.environment.get_grid('cost')
        potency = self.environment.get_grid('potency')
        
        # Sort on-fire cells based on cost and potency (lower cost and higher potency -> more likely to suppress)
        sorted_on_fire_cells = sorted(on_fire_cells, key=lambda x: (cost[tuple(x)], -potency[tuple(x)]))
        
        # Distribute suppressions
        for cell in sorted_on_fire_cells[:num_suppressions]:
            x, y = cell
            self.environment.grid[x, y] = GridState.SUPPRESSED.value
            self.updated_cells.add(cell)

    def simulate_step(self):
        # Get the current population (number of cells on fire)
        current_population = np.sum((self.environment.grid == GridState.ON_FIRE.value) | (self.environment.grid == GridState.BURNED.value))
        
        # Get the current footprint (total area affected by fire)
        current_footprint = current_population + np.sum((self.environment.grid == GridState.PREVIOUSLY_BURNED.value) | (self.environment.grid == GridState.SUPPRESSED.value))

        # update ON_FIRE to BURNED
        self.environment.grid[self.environment.grid == GridState.ON_FIRE.value] = GridState.BURNED.value  

        # create a log statement with the full sim info
        print(f'COMBINED:\nPopulation: {current_population}, Footprint: {current_footprint}\n\n')
        
        # Simulate one time step using the temporal model
        next_population, next_footprint, firelets, extinguishments, suppressions = self.wildfire_model._simulate_step(
            current_population, current_footprint)
        
        # Update the spatial model based on the new firelets, extinguishments, and suppressions
        self._distribute_firelets(firelets)
        self._distribute_extinguishments(extinguishments)
        self._distribute_suppressions(suppressions)
        self.updated_cells = set()

        # update the environment to account for changes
        self.environment.update()

        # create a log statement with the full sim info
        print(f'COMBINED:\nFirelets: {firelets}, Extinguishments: {extinguishments}, Suppressions: {suppressions}, Next Population: {next_population}, Next Footprint: {next_footprint}\n\n')
        
        
        return next_population, next_footprint