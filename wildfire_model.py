import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable

class WildfireSpreadProcess:
    """
    This class represents a continuous-time Markov process simulating the spread of wildfires.
    
    The class is initialized with three parameters:
    - spread_rate: Either a constant, or a function that takes the current population as an argument and returns the spread rate.
    - extinguish_rate: Either a constant, or a function that takes the current population as an argument and returns the extinguish rate.
    - firefighting_rate: Either a constant, or a function that takes the current population as an argument and returns the firefighting rate.
    - system_size: The total burnable area.
    """
    def __init__(self, spread_rate: Union[float, Callable[[int], float]], extinguish_rate: Union[float, Callable[[int], float]], firefighting_rate: Union[float, Callable[[int], float]], system_size: int):
        self.spread_rate = spread_rate
        self.extinguish_rate = extinguish_rate
        self.firefighting_rate = firefighting_rate
        self.system_size = system_size

    def get_rate(self, rate, population):
        """
        Return the rate, which can be either a constant or a function of the population.

        Parameters:
        - rate: Either a constant, or a function that takes the current population as an argument and returns the rate.
        - population: The current population.

        Returns:
        - The rate for the current population.
        """
        if callable(rate):
            return rate(population)
        else:
            return rate

    def simulate(self, initial_population: int, time_steps: int):
        """
        Simulate the wildfire spread process over a given number of time steps.

        Parameters:
        - initial_population: The starting size of the wildfire.
        - time_steps: The number of time steps over which to simulate the process.

        Returns:
        - A list representing the size of the wildfire at each time step.
        - A list representing the burned area at each time step.
        - The time step at which the fire was extinguished.
        """
        # Initialize the population list with the initial population
        population = [initial_population]
        footprint = [initial_population]  # burned area
        extinguishment_time = None  # start with no extinguishment time

        for t in range(time_steps):
            # Calculate the spread, natural extinguish and suppression
            spread = np.random.poisson(self.get_rate(self.spread_rate, population[-1]) * population[-1])
            extinguish = np.random.poisson(self.get_rate(self.extinguish_rate, population[-1]) * population[-1])
            suppression = np.random.poisson(self.get_rate(self.firefighting_rate, population[-1]) * population[-1])

            # Ensure the spread does not exceed the unburned area
            spread = min(spread, self.system_size - footprint[-1])
            
            # Ensure the population never goes below 0 and above the system size
            next_population = min(max(population[-1] + spread - extinguish - suppression, 0), self.system_size)
            next_footprint = min(footprint[-1] + spread, self.system_size)

            # Append the next population to the list
            population.append(next_population)
            footprint.append(next_footprint)

            # Record the extinguishment time
            if next_population == 0 and extinguishment_time is None:
                extinguishment_time = t

        return population, footprint, extinguishment_time

if __name__ == '__main__':
    # Function that calculates spread rate depending on the current population
    def spread_rate_fn(population):
        return 0.05 if population < 100 else 0.005
    
    # Function that calculates extinguish rate depending on the current population
    def extinguish_rate_fn(population):
        return 0.05 if population < 100 else 0.1

    # Function that calculates firefighting rate depending on the current population
    def firefighting_rate_fn(population):
        return 0.0 if population < 100 else 0.1
    
    # Create a wildfire spread process with spread, extinguish and firefighting rates that depend on the current population, and a system size of 1000
    ws_process = WildfireSpreadProcess(spread_rate_fn, extinguish_rate_fn, firefighting_rate_fn, 1000)
    
    # Simulate the process over 1000 time steps with an initial population of 100 individuals
    population, footprint, extinguishment_time = ws_process.simulate(100, 1000)

    # Plot the population and footprint over time
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Active Fire', color=color)
    ax1.plot(population, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Burned Area', color=color)  # we already handled the x-label with ax1
    ax2.plot(footprint, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Wildfire Spread Process Example')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # add text for the extinguishment time
    if extinguishment_time is not None:
        plt.axvline(x=extinguishment_time, color='r', linestyle='--')

    plt.show()

    # Print the extinguishment time
    if extinguishment_time is not None:
        print(f"The fire was extinguished at time step {extinguishment_time}.")
    else:
        print("The fire was not extinguished within the simulation time.")
