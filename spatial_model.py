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
    FIREBREAK = 5
    REMOVED = 6  # State for removed trees due to thinning
    SUPPRESSED = 7  # State for suppressed fires
    PREVIOUSLY_SUPPRESSED = 8  # State for previously suppressed fires

class Environment:
    def __init__(self, perturbations: List[Tuple[Tuple[int, int], float]]):
        self.perturbations = perturbations
        self.wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        self.budget = 100.0

    def get_perturbation(self, grid: np.ndarray=None) -> Tuple[int, int]:
        """add a perturbation to the grid at a random location with size initial_population"""
        if grid is not None:
            tree_mask = (grid == GridState.TREE.value)
            dist = self.get_fire_proximity(grid) 
            perturbations = [((i,j), dist[i,j]) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if tree_mask[i,j]] 
            self.perturbations = perturbations
        return random.choices(
            population=[p[0] for p in self.perturbations],
            weights=[p[1] for p in self.perturbations],
            k=1
        )[0]

    def get_wind_direction(self) -> str:
        return random.choice(self.wind_directions)

    def get_fire_proximity(self, grid: np.ndarray) -> np.ndarray:
        fire_coords = np.argwhere(grid == GridState.ON_FIRE.value)
        if len(fire_coords) == 0:
            return np.zeros(grid.shape)

        dist = np.zeros(grid.shape)
        for fire in fire_coords:
            distances = np.sum((fire_coords - fire) ** 2, axis=1)
            dist[fire[0], fire[1]] = np.sum(np.exp(-distances / (2**2)))

        # Normalize the distances to get probabilities
        dist = dist / np.sum(dist)

        # Replace nan with a very small probability
        dist = np.nan_to_num(dist, nan=1e-5)

        return dist

    def calculate_spread_rate(self, topography: Tuple[float, float], 
                              wind: Tuple[float, float], 
                              fuel_conditions: Tuple[float, str],
                              atmospheric_conditions: Tuple[float, float], fire_proximity: float) -> float:
        slope, vegetation_density = topography
        wind_vector = np.array(wind)
        moisture, fuel_type = fuel_conditions
        humidity, temperature = atmospheric_conditions

        directional_speed = np.dot(wind_vector, np.array([np.sin(slope), np.cos(slope)]))
        fuel_factor = self.fuel_factor(vegetation_density, moisture, fuel_type)
        atmospheric_factor = self.atmospheric_factor(humidity, temperature)

        return directional_speed * fuel_factor * atmospheric_factor * (1 + fire_proximity)

    def calculate_extinguish_rate(self, natural_barriers: float, 
                                  weather_conditions: float, extinguish_probability: float) -> float:
        return self.environment_factor(natural_barriers, weather_conditions) * extinguish_probability
    
    def calculate_firefighting_rate(self, current_phase: str, 
                                     mobility: float, 
                                     potency: float, 
                                     cost: float) -> float:
        base_rate = self.resource_allocation(current_phase, mobility, potency, cost)
        capped_rate = min(base_rate, self.budget)
        self.budget -= capped_rate  # Update remaining budget
        return capped_rate

    @staticmethod
    def fuel_factor(vegetation_density: float, moisture: float, fuel_type: str) -> float:
        # Assuming vegetation_density is a factor between 0 and 1, where 1 means very dense vegetation
        # Assuming moisture is a percentage between 0 and 100
        # Assuming fuel_type could be "grass", "brush", "timber", etc.

        # Moisture effect: The wetter, the less conducive for fire
        moisture_factor = 1 - (moisture / 100.0)

        # Vegetation density effect
        density_factor = vegetation_density

        # Fuel type effect
        fuel_type_factor = {
            'grass': 0.6,
            'brush': 0.8,
            'timber': 1.0
        }.get(fuel_type, 0.5)  # default to 0.5 if fuel type is unknown

        return min(max(moisture_factor * density_factor * fuel_type_factor, 0), 1)

    @staticmethod
    def atmospheric_factor(humidity: float, temperature: float) -> float:
        # Assuming humidity is a percentage between 0 and 100
        # Assuming temperature is in degrees Celsius
        # The atmospheric factor ranges between 0 and 1
        humidity_factor = 1 - (humidity / 100.0)
        temperature_factor = (temperature - 20) / 30.0 if temperature > 20 else 0
        return min(max(humidity_factor + temperature_factor, 0), 1)
    
    @staticmethod
    def environment_factor(natural_barriers: float, weather_conditions: float) -> float:
        # Assuming natural_barriers is a factor between 0 and 1 where 1 means many natural barriers
        # Assuming weather_conditions is a factor between 0 and 1 where 1 means favorable weather
        return min(natural_barriers + weather_conditions, 1)

    
    @staticmethod
    def resource_allocation(current_phase: str, mobility: float, potency: float, cost: float) -> float:
        # Assuming current_phase could be "initial", "developed", or "controlled"
        # Assuming mobility, potency, and cost are factors between 0 and 1

        phase_factor = {
            'initial': 1,
            'developed': 0.8,
            'controlled': 0.5
        }.get(current_phase, 0.5)  # default to 0.5 if phase is unknown

        resource_factor = mobility * 0.4 + potency * 0.4 + cost * 0.2
        return min(phase_factor * resource_factor, 1)


class System:
    def __init__(self, grid: np.ndarray):
        self.grid = grid

    def get_perimeter_cells(self, grid: np.ndarray):
        """get the cells at the perimeter of the fire"""
        perimeter_cells = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == GridState.ON_FIRE.value:
                    #print(f"Cell ({i}, {j}) is on fire")
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                            if grid[ni, nj] == GridState.TREE.value:
                                perimeter_cells.append((i, j))
                                break
        return perimeter_cells

    def evaluate_performance(self) -> float:
        tree_density = np.sum(self.grid) / (self.grid.shape[0] * self.grid.shape[1])
        return tree_density

class Optimization:
    def __init__(self, system: System, environment: Environment, resources: List[Tuple[str, float]]):
        self.system = system
        self.environment = environment
        self.resources = resources

    def apply_resources(self, grid: np.ndarray, resources: List[Tuple[str, float]]) -> np.ndarray:
        for resource in resources:
            resource_type, resource_strength = resource
            if resource_strength < np.random.uniform():
                return grid
            if resource_type == 'firebreak':
                previously_burned = np.argwhere(grid == GridState.PREVIOUSLY_BURNED.value)
                for i, j in previously_burned:
                    if i > 0 and grid[i-1, j] == GridState.TREE.value:
                        grid[i-1, j] = GridState.FIREBREAK.value
            elif resource_type == 'thinning':
                # Similar to the original code but could be extended to more sophisticated methods
                pass  # Placeholder
        return grid

class Simulation:
    def __init__(self, system: System, environment: Environment, resources: List[Tuple[str, float]]):
        self.system = system
        self.environment = environment
        self.resources = resources

    def simulate_step(self, grid: np.ndarray) -> np.ndarray:
        new_grid = grid.copy()
        wind_direction = self.environment.get_wind_direction()
        extinguish_probability = self.environment.get_fire_proximity(grid)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == GridState.ON_FIRE.value:
                    # Cell is on fire

                    # Extinguishing fire based on probability
                    if np.random.uniform() < extinguish_probability[i, j]:
                        new_grid[i, j] = GridState.BURNED.value
                        continue

                    if i > 0 and grid[i-1, j] == GridState.TREE.value and wind_direction in ['N', 'NW', 'NE']:
                        new_grid[i-1, j] = GridState.ON_FIRE.value # Spread fire to north 
                    if i < grid.shape[0]-1 and grid[i+1, j] == GridState.TREE.value and wind_direction in ['S', 'SW', 'SE']:
                        new_grid[i+1, j] = GridState.ON_FIRE.value  # Spread fire to south
                    if j > 0 and grid[i, j-1] == GridState.TREE.value and wind_direction in ['W', 'NW', 'SW']:
                        new_grid[i, j-1] = GridState.ON_FIRE.value # Spread fire to west
                    if j < grid.shape[1]-1 and grid[i, j+1] == GridState.TREE.value and wind_direction in ['E', 'NE', 'SE']:
                        new_grid[i, j+1] = GridState.ON_FIRE.value # Spread fire to east
                elif grid[i, j] == GridState.BURNED.value:
                    # Cell is burned
                    new_grid[i, j] = GridState.PREVIOUSLY_BURNED.value
        return new_grid

    def run_simulation(self, initial_population: int, time_steps: int):
        population, footprint, extinguishment_time = self.temporal_model.simulate(initial_population, time_steps)

        # Initialize System and Optimization classes
        system = System(self.grid)
        optimization = Optimization(system, self.environment, self.resources)

        grids = [self.grid]
        for t in range(1, len(population)):
            new_fire_cells = population[t] - population[t-1]

            # Apply resources before updating the grid
            optimized_system = optimization.optimize()

            new_grid = self.update_grid(optimized_system.grid, new_fire_cells)  # Use the optimized grid
            grids.append(new_grid)

        return grids, extinguishment_time

    def analyze_results(self, grid: np.ndarray) -> float:
        tree_density = np.sum(grid) / (grid.shape[0] * grid.shape[1])
        return tree_density
