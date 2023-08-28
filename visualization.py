import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from combined_model import CombinedModel
from temporal_model import WildfireSpreadProcess
from spatial_model import GridState, Environment
import numpy as np

def run_visualization(grid_size, n_steps, initial_population, time_steps):
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
    combined_model = CombinedModel(temporal_model, grid, environment)

    # Run simulation
    grids = combined_model.run_simulation(initial_population, time_steps)

    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_title('Forest Fire Simulation')
    cmap = ListedColormap(['white', 'green', 'red', 'grey'])

    def update(frame):
        ax.clear()
        ax.set_title(f'Forest Fire Simulation (Time Step: {frame+1})')
        im = ax.imshow(grids[frame], cmap=cmap, vmin=0, vmax=3)
        ax.set_xticks([])
        ax.set_yticks([])

    ani = FuncAnimation(fig, update, frames=len(grids), interval=1000, blit=False)
    plt.show()

if __name__ == "__main__":
    run_visualization(grid_size=50, n_steps=50, initial_population=10, time_steps=100)
