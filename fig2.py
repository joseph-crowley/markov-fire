import matplotlib.pyplot as plt
from wildfire_model import WildfireSpreadProcess

# The birth and death rates are equal
birth_death_rate = 0.05  # example rate

# Create a wildfire spread process with the same birth (spread) and death (extinguish) rates, and a system size of 1000
ws_process = WildfireSpreadProcess(birth_death_rate, birth_death_rate, 0, 1000)

# Simulate the process over 1000 time steps with an initial population of 10
population, footprint, extinguishment_time = ws_process.simulate(10, 1000)

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
plt.show()

# Print the extinguishment time
if extinguishment_time is not None:
    print(f"The fire was extinguished at time step {extinguishment_time}.")
else:
    print("The fire was not extinguished within the simulation time.")
