import matplotlib.pyplot as plt
import numpy as np
from wildfire_model import WildfireSpreadProcess

# Parameters
birth_death_rate = 1  # birth rate = death rate
initial_population = 100
time_steps = 1000
n_simulations = 1000  # number of Monte Carlo simulations

# Store the results of each simulation
results = np.zeros((n_simulations, time_steps + 1))
extinguishment_times = np.full(n_simulations, time_steps)  # Initialize to the maximum time

# Run the Monte Carlo simulations
for i in range(n_simulations):
    ws_process = WildfireSpreadProcess(birth_death_rate, birth_death_rate, 0, 1000)
    population, _, extinguishment_time = ws_process.simulate(initial_population, time_steps)
    results[i] = population
    if extinguishment_time is not None:  # If the fire was extinguished
        extinguishment_times[i] = extinguishment_time


# Calculate the variance of the population at each time step
population_variance = np.var(results, axis=0)

# Set the variance to 0 for all time steps after the fire was extinguished
for t in range(time_steps + 1):
    if np.mean(extinguishment_times <= t):  # If the fire was extinguished on average
        population_variance[t:] = 0
        break

# Plot the variance over time
plt.plot(population_variance, label='Variance')
plt.xlim(0, np.max(extinguishment_times))

plt.title(f'Population Variance Over Time\nBirth Rate = Death Rate = {birth_death_rate}, Initial Population = {initial_population}')
plt.xlabel('Time Step')
plt.ylabel('Population Variance')
plt.legend()

# Add a textbox with the parameters
textstr = '\n'.join((
    r'$\beta=\delta=%.2f$' % (birth_death_rate, ),
    r'$N=%i$' % (initial_population, ),
    r'$timesteps=%i$' % (time_steps, ),
    r'$simulations=%i$' % (n_simulations, )))
plt.text(0.75, 0.35, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()
