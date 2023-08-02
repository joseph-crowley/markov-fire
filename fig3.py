import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_population = 5
time_steps = 50
beta_values = [0.8, 1, 1.2]

# Time array
time = np.arange(time_steps + 1)

# Calculate and plot the absorption probability for each beta
p_A_max = 0
for beta in beta_values:
    if beta == 1:
        p_A = (time / (1 + time))**initial_population
    else:
        p_A = (1 - np.exp((beta - 1) * time)) / (1 - beta * np.exp((beta - 1) * time))**initial_population

    # Calculate cumulative sum of p_A
    p_A = np.cumsum(p_A)
    
    # Normalize p_A so that its maximum value is 1
    if np.max(p_A) > p_A_max:
        p_A_max = np.max(p_A)

    p_A = p_A / p_A_max 

    plt.plot(time, p_A, label=f'Beta = {beta}')

# Set up the plot
plt.title(f'Cumulative Absorption Probability\nInitial Population = {initial_population}')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Absorption Probability')
plt.legend()

plt.show()
