import numpy as np
from typing import Union, Callable, Tuple, List

class WildfireSpreadModel:
    """
    A class to model the spread of wildfire over time.
    
    Attributes:
        spread_rate (Union[float, Callable]): The rate at which the wildfire spreads.
        extinguish_rate (Union[float, Callable]): The rate at which the wildfire extinguishes naturally.
        firefighting_rate (Union[float, Callable]): The rate at which firefighting efforts suppress the wildfire.
        system_size (int): The maximum size of the area that can be affected by wildfire.
    """
    
    def __init__(self, 
                 spread_rate: Union[float, Callable[[int], float]], 
                 extinguish_rate: Union[float, Callable[[int], float]], 
                 firefighting_rate: Union[float, Callable[[int], float]], 
                 system_size: int) -> None:
        """Initialize the wildfire model with rates and system size."""
        self.spread_rate = spread_rate
        self.extinguish_rate = extinguish_rate
        self.firefighting_rate = firefighting_rate
        self.system_size = system_size

    def _get_rate(self, rate: Union[float, Callable[[int], float]], population: int) -> float:
        """Get the rate, accounting for callable rates that depend on the population."""
        return rate(population) if callable(rate) else rate

    def _simulate_step(self, current_population: int, current_footprint: int) -> Tuple[int, int, int, int, int]:
        """Simulate one time step of the wildfire spread."""
        print(f'TEMPORAL:\nCalling _simulate_step with current_population: {current_population}, current_footprint: {current_footprint}\n')
        
        spread = np.random.poisson(self._get_rate(self.spread_rate, current_population) * current_population)
        extinguish = np.random.poisson(self._get_rate(self.extinguish_rate, current_population) * current_population)
        suppression = np.random.poisson(self._get_rate(self.firefighting_rate, current_population) * current_population)

        spread = min(spread, self.system_size - current_footprint)
        
        next_population = max(min(current_population + spread - extinguish - suppression, self.system_size), 0)
        next_footprint = min(current_footprint + spread, self.system_size)

        # create a log statement with the full sim info including rates and features
        print(f'TEMPORAL:\n Rates: {self.spread_rate}, {self.extinguish_rate}, {self.firefighting_rate}, Firelets: {spread}, Extinguishments: {extinguish}, Suppressions: {suppression}, Population: {next_population}, Footprint: {next_footprint}\n\n')
        
        return next_population, next_footprint, spread, extinguish, suppression

    def simulate(self, initial_population: int, time_steps: int) -> Tuple[List[int], List[int], int, List[int], List[int], List[int]]:
        """
        Simulate the wildfire spread for a given number of time steps.
        
        Returns:
            population (List[int]): The population affected by the wildfire at each time step.
            footprint (List[int]): The total area affected by the wildfire at each time step.
            extinguishment_time (int): The time step at which the fire was extinguished.
            spread_counts (List[int]): The number of new areas affected by the wildfire at each time step.
            extinguish_counts (List[int]): The number of areas where the wildfire was extinguished at each time step.
            suppression_counts (List[int]): The number of areas where the wildfire was suppressed at each time step.
        """
        population = [initial_population]
        footprint = [initial_population]
        extinguishment_time = None

        spread_counts = []
        extinguish_counts = []
        suppression_counts = []
        
        for t in range(time_steps):
            next_population, next_footprint, spread, extinguish, suppression = self._simulate_step(
                population[-1], footprint[-1]
            )
            
            population.append(next_population)
            footprint.append(next_footprint)
            spread_counts.append(spread)
            extinguish_counts.append(extinguish)
            suppression_counts.append(suppression)

            if next_population == 0 and extinguishment_time is None:
                extinguishment_time = t

        return population, footprint, extinguishment_time, spread_counts, extinguish_counts, suppression_counts
