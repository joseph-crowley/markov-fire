import numpy as np
from typing import Union, Callable

class WildfireSpreadProcess:
    def __init__(self, 
                 spread_rate: Union[float, Callable[[int], float]], 
                 extinguish_rate: Union[float, Callable[[int], float]], 
                 firefighting_rate: Union[float, Callable[[int], float]], 
                 system_size: int):
        self.spread_rate = spread_rate
        self.extinguish_rate = extinguish_rate
        self.firefighting_rate = firefighting_rate
        self.system_size = system_size

    def get_rate(self, rate, population):
        if callable(rate):
            return rate(population)
        else:
            return rate

    def simulate(self, initial_population: int, time_steps: int):
        # Initialize the population and footprint lists
        population = [initial_population]
        footprint = [initial_population]
        extinguishment_time = None  # Keep track of when the fire is extinguished
        
        for t in range(time_steps):
            # Calculate spread, extinguish, and suppression using variable rates if applicable
            spread = np.random.poisson(self.get_rate(self.spread_rate, population[-1]) * population[-1])
            extinguish = np.random.poisson(self.get_rate(self.extinguish_rate, population[-1]) * population[-1])
            suppression = np.random.poisson(self.get_rate(self.firefighting_rate, population[-1]) * population[-1])
            
            # Ensure the spread does not exceed the unburned area
            spread = min(spread, self.system_size - footprint[-1])
            
            # Calculate the next population and footprint
            next_population = min(max(population[-1] + spread - extinguish - suppression, 0), self.system_size)
            next_footprint = min(footprint[-1] + spread, self.system_size)
            
            # Append the next population and footprint to the list
            population.append(next_population)
            footprint.append(next_footprint)
            
            # Update the extinguishment time if the fire is out
            if next_population == 0 and extinguishment_time is None:
                extinguishment_time = t

        return population, footprint, extinguishment_time
