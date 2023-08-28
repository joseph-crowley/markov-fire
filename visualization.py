import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from combined_model import CombinedModel
from temporal_model import WildfireSpreadProcess
from spatial_model import GridState, Environment, System, Optimization, Simulation
import numpy as np

def run_visualization(grid_size, n_steps, initial_population, time_steps, resources):
    # Initialize temporal model
    spread_rate = 0.05
    extinguish_rate = 0.01
    firefighting_rate = 0.02
    system_size = grid_size ** 2
    temporal_model = WildfireSpreadProcess(spread_rate, extinguish_rate, firefighting_rate, system_size)

    # Initialize spatial model
    perturbations = [((i, j), 1.0) for i in range(grid_size) for j in range(grid_size)]
    environment = Environment(perturbations)

    # Initialize grid
    grid = np.random.choice([GridState.EMPTY.value, GridState.TREE.value], size=(grid_size, grid_size), p=[0.1, 0.9])

    # Initialize combined model
    combined_model = CombinedModel(temporal_model, grid, environment, resources)
 
    # Run simulation
    grids, extinguishment_time = combined_model.run_simulation(initial_population, time_steps)
    populations = [np.sum(g == GridState.ON_FIRE.value) for g in grids]
    footprints = [np.sum((g == GridState.ON_FIRE.value) | (g == GridState.BURNED.value) | (g == GridState.PREVIOUSLY_BURNED.value)) for g in grids]
    
    # First Plot: Grid Animation
    fig1, ax1 = plt.subplots()
    ax1.set_title('Forest Fire Simulation')
    cmap = ListedColormap(['white', 'green', 'red', 'grey', 'black'])

    def update(frame):
        ax1.clear()
        ax1.set_title(f'Forest Fire Simulation (Time Step: {frame+1})')
        im = ax1.imshow(grids[frame], cmap=cmap, vmin=0, vmax=4)
        ax1.set_xticks([])
        ax1.set_yticks([])

    ani = FuncAnimation(fig1, update, frames=len(grids), interval=1000, blit=False)
    
    # Second Plot: Population and Footprint
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Active Fire', color='tab:red')
    ax2.plot(populations, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax3 = ax2.twinx()
    ax3.set_ylabel('Burned Area', color='tab:blue')
    ax3.plot(footprints, color='tab:blue')
    ax3.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title('Wildfire Spread Process Example')
    fig2.tight_layout()

    plt.show()

    # Print the extinguishment time
    if extinguishment_time is not None:
        print(f"The fire was extinguished at time step {extinguishment_time}.")
    else:
        print("The fire was not extinguished within the simulation time.")

if __name__ == "__main__":
    resources = [("firebreak", 0.5), ("thinning", 0.3)]  # Example resources
    run_visualization(grid_size=50, n_steps=50, initial_population=10, time_steps=100, resources=resources)
