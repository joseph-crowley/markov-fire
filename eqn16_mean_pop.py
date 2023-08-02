import matplotlib.pyplot as plt
import numpy as np
from wildfire_model import WildfireSpreadProcess

# Parameters
birth_rate = 0.09  # example rate
death_rate = 0.07
initial_population = 10
time_steps = 200
n_simulations = 1000  # number of Monte Carlo simulations

# Store the results of each simulation
results = np.zeros((n_simulations, time_steps + 1))

# Run the Monte Carlo simulations
for i in range(n_simulations):
    ws_process = WildfireSpreadProcess(birth_rate, death_rate, 0, 1000)
    population, _, _ = ws_process.simulate(initial_population, time_steps)
    results[i] = population

# Calculate the mean population at each time step
mean_population = np.mean(results, axis=0)

# Calculate the theoretical mean population at each time step
theoretical_mean_population = initial_population * np.exp((birth_rate - death_rate) * np.arange(time_steps + 1))

# Plot the simulated and theoretical mean population over time
plt.plot(mean_population, label='Simulated')
plt.plot(theoretical_mean_population, label='Theoretical')

# add to title whether beta > delta or beta < delta
str_to_add = ''
if birth_rate > death_rate:
    str_to_add = ' (beta > delta)'
elif birth_rate < death_rate:
    str_to_add = ' (beta < delta)'

plt.title('Mean Population Over Time' + str_to_add)
plt.xlabel('Time Step')
plt.ylabel('Mean Population')
plt.yscale('log')
plt.legend()

# Add a textbox with the parameters
textstr = '\n'.join((
    r'$\beta=%.2f$' % (birth_rate, ),
    r'$\delta=%.2f$' % (death_rate, ),
    r'$N=%i$' % (initial_population, ),
    r'$timesteps=%i$' % (time_steps, ),
    r'$simulations=%i$' % (n_simulations, )))
plt.text(0.75, 0.35, textstr, transform=plt.gca().transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


plt.show()
