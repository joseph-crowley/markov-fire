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
        population, footprint, extinguishment_time, spread_counts, extinguish_counts, suppression_counts = self.temporal_model.simulate(initial_population, time_steps)

        # Initialize the spatial simulation
        grids = [self.initial_grid]
        
        for t in range(1, len(population)):
            # Unpack the spread, extinguish, and suppression counts for the current time step
            new_fire_cells = spread_counts[t] if t < len(spread_counts) else 0
            extinguished_cells = extinguish_counts[t] if t < len(extinguish_counts) else 0
            suppressed_cells = suppression_counts[t] if t < len(suppression_counts) else 0
            
            # Update the grid based on these counts
            new_grid = self.update_grid(grids[-1], new_fire_cells, extinguished_cells, suppressed_cells)
            grids.append(new_grid)

        return grids, extinguishment_time

    def update_grid(self, grid: np.ndarray, new_fire_cells: int, extinguished_cells: int, suppressed_cells: int):
        # Apply resources before updating the fire
        optimized_grid = self.optimization.apply_resources(grid, self.resources)

        # calculate effective rates for each cell
        effective_rates = self.calculate_effective_rates(optimized_grid)
        
        #print("\n\nEffective rates (update grid):")
        #print(effective_rates)

        # Update the optimized_grid based on new_fire_cells, extinguished_cells, suppressed_cells
        # This part can be implemented in various ways. For example, you could randomly select cells to update,
        # or you could use some sort of intelligent mapping between the counts and the spatial grid.
        # For simplicity, let's assume we have a function that does this: `update_cells()`
        optimized_grid = self.update_cells(optimized_grid, new_fire_cells, extinguished_cells, suppressed_cells, effective_rates)

        # Run the spatial simulation for one step
        new_grid = self.simulation.simulate_step(optimized_grid)

        return new_grid

    def update_cells(self, grid, new_fire_cells, extinguished_cells, suppressed_cells, effective_rates):
        new_grid = grid.copy()

        # Index candidate cells for each type of state change
        candidate_fire_cells = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == GridState.TREE.value]
        candidate_extinguish_cells = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i, j] == GridState.ON_FIRE.value]
        candidate_suppress_cells = candidate_extinguish_cells  # Assuming the same cells can be suppressed

        # Calculate weights based on effective_rates
        fire_weights = [effective_rates.get(cell, {}).get('spread', 0) for cell in candidate_fire_cells]
        extinguish_weights = [effective_rates.get(cell, {}).get('extinguish', 0) for cell in candidate_extinguish_cells]
        suppress_weights = [effective_rates.get(cell, {}).get('suppress', 0) for cell in candidate_suppress_cells]
    
        #print("\n\nWeights (update cells):")
        #print(fire_weights)
        #print(extinguish_weights)
        #print(suppress_weights)

        # Randomly select cells to update based on weights (using indices)
        if fire_weights:
            selected_fire_indices = np.random.choice(len(candidate_fire_cells), size=new_fire_cells, replace=False, p=normalize_weights(fire_weights))
        else:
            selected_fire_indices = []

        if extinguish_weights:
            selected_extinguish_indices = np.random.choice(len(candidate_extinguish_cells), size=extinguished_cells, replace=False, p=normalize_weights(extinguish_weights))
        else:
            selected_extinguish_indices = []

        if suppress_weights:
            selected_suppress_indices = np.random.choice(len(candidate_suppress_cells), size=suppressed_cells, replace=False, p=normalize_weights(suppress_weights))
        else:
            selected_suppress_indices = []

        #print("\n\nSelected indices (update cells):")
        #print(selected_fire_indices)
        #print(selected_extinguish_indices)
        #print(selected_suppress_indices)

        # Map indices back to (i, j) tuples
        new_fires = [candidate_fire_cells[i] for i in selected_fire_indices]
        extinguishments = [candidate_extinguish_cells[i] for i in selected_extinguish_indices]
        suppressions = [candidate_suppress_cells[i] for i in selected_suppress_indices]

        # Update the grid
        for i, j in new_fires:
            new_grid[i, j] = GridState.ON_FIRE.value
        for i, j in extinguishments:
            new_grid[i, j] = GridState.BURNED.value
        for i, j in suppressions:
            new_grid[i, j] = GridState.REMOVED.value  # Or some other state representing suppression

        return new_grid


    def calculate_effective_rates(self, grid: np.ndarray):
        effective_rates = {}
        perimeter_cells = self.system.get_perimeter_cells(grid)
        #print("\n\nPerimeter cells (calculate effective rates):")
        #print(perimeter_cells)
        n = len(perimeter_cells)

        for cell in perimeter_cells:
            i, j = cell
            # Fetch real-time data for this cell (dummy values for now)
            topography = (0.1, 0.9)  # slope, vegetation_density
            wind = (1.0, 0.5)  # wind vector
            fuel_conditions = (0.2, 'grass')  # moisture, fuel_type
            atmospheric_conditions = (0.7, 25.0)  # humidity, temperature
            natural_barriers = 0.1
            weather_conditions = 0.1
            current_phase = 'initial'
            mobility = 1.0
            potency = 1.0
            cost = 1.0

            # Calculate rates for this cell
            spread_rate = self.environment.calculate_spread_rate(topography, wind, fuel_conditions, atmospheric_conditions)
            extinguish_rate = self.environment.calculate_extinguish_rate(natural_barriers, weather_conditions)
            firefighting_rate = self.environment.calculate_firefighting_rate(current_phase, mobility, potency, cost)

            # Store effective rates for this cell
            effective_rates[cell] = {
                'spread': spread_rate / n if n > 0 else 0,
                'extinguish': extinguish_rate / n if n > 0 else 0,
                'suppress': firefighting_rate / n if n > 0 else 0
            }

        return effective_rates


def normalize_weights(weights):
    #print("\n\nWeights (normalize weights):")
    #print(weights)

    total = sum(weights)
    if total == 0:
        return [1 / len(weights) for _ in weights]
    normalized_weights = [w / total for w in weights]
    epsilon = 1e-10
    normalized_weights = [w + epsilon for w in normalized_weights]
    sum_weights = sum(normalized_weights)
    return [w / sum_weights for w in normalized_weights]



