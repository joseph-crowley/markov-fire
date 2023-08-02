import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable

class BirthDeathProcess:
    """
    This class represents a continuous-time Markov process known as a birth-death process.
    
    The class is initialized with two parameters:
    - birth_rate: Either a constant, or a function that takes the current population as an argument and returns the birth rate.
    - death_rate: Either a constant, or a function that takes the current population as an argument and returns the death rate.
    """
    def __init__(self, birth_rate: Union[float, Callable[[int], float]], death_rate: Union[float, Callable[[int], float]]):
        self.birth_rate = birth_rate
        self.death_rate = death_rate

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
        Simulate the birth-death process over a given number of time steps.

        Parameters:
        - initial_population: The starting number of individuals in the population.
        - time_steps: The number of time steps over which to simulate the process.

        Returns:
        - A list representing the number of individuals in the population at each time step.
        """
        # Initialize the population list with the initial population
        population = [initial_population]

        for _ in range(time_steps):
            # Calculate the number of births and deaths
            births = np.random.poisson(self.get_rate(self.birth_rate, population[-1]) * population[-1])
            deaths = np.random.poisson(self.get_rate(self.death_rate, population[-1]) * population[-1])

            # Ensure the population never goes below 0
            next_population = max(population[-1] + births - deaths, 0)

            # Append the next population to the list
            population.append(next_population)

        return population

if __name__ == '__main__':
    # Function that calculates birth rate depending on the current population
    def birth_rate_fn(population):
        return 0.01 if population < 100 else 0.005
    
    # Function that calculates death rate depending on the current population
    def death_rate_fn(population):
        return 0.005 if population < 100 else 0.01
    
    # Create a birth-death process with birth and death rates that depend on the current population
    bd_process = BirthDeathProcess(birth_rate_fn, death_rate_fn)
    
    # Simulate the process over 1000 time steps with an initial population of 100 individuals
    population = bd_process.simulate(100, 1000)
    
    # Plot the population over time
    plt.plot(population)
    plt.title('Birth-Death Process Example')
    plt.xlabel('Time Step')
    plt.ylabel('Population')
    plt.show()
    