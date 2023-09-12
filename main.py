from combined_model import CombinedModel  
from spatial_dynamics import Environment  
from temporal_dynamics import WildfireSpreadModel  
from utils import GridState

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import numpy as np

def initialize_grid(grid_size: int, initial_population: int, environment: Environment):
    """Initialize the grid with empty and tree cells, and add initial fire."""
    environment.grid = np.random.choice([GridState.EMPTY.value, GridState.TREE.value], size=(grid_size, grid_size), p=[0.1, 0.9])
    fire_location = np.random.randint(0, grid_size, size=2)
    for i in range(initial_population):
        firelet_location = fire_location + np.random.randint(-2, 3, size=2)
        firelet_location = np.clip(firelet_location, 0, grid_size-1)
        environment.grid[firelet_location[0], firelet_location[1]] = GridState.ON_FIRE.value

def run_simulation(grid_size: int, initial_population: int, time_steps: int):
    """Run the wildfire simulation and return grids, populations, and footprints."""
    # Initialize models
    system_size = grid_size ** 2
    
    # Initialize the environment and grid (example with defaults)
    environment = Environment(np.zeros((grid_size, grid_size), dtype=int))

    initialize_grid(grid_size, initial_population, environment)
    environment.set_default_values() # replace with real-world data

    # initialize rates (TODO: Calculate specific rates based on environment)
    spread_rate = 0.05
    extinguish_rate = 0.05
    suppression_rate = 0.01
    
    # Initialize the combined model
    wildfire_model = WildfireSpreadModel(spread_rate, extinguish_rate, suppression_rate, system_size)
    combined_model = CombinedModel(wildfire_model, environment)

    current_population = np.sum((combined_model.environment.grid == GridState.ON_FIRE.value) | (combined_model.environment.grid == GridState.BURNED.value))

    print(f'MAIN:\nCurrent Population: {current_population}\n')

    grids = []
    populations = []
    footprints = []
    extinguishment_time = None
    for t in range(time_steps):
        # Simulate one step
        next_population, next_footprint = combined_model.simulate_step()

        # Save the grid, population, and footprint
        grids.append(combined_model.environment.grid.copy())
        populations.append(next_population)
        footprints.append(next_footprint)

        if next_population == 0:
            extinguishment_time = t + 1
            break
    
    return grids, populations, footprints, extinguishment_time

def visualize_simulation(grids: np.ndarray, populations: np.ndarray, footprints: np.ndarray):
    """Visualize the simulation using Matplotlib."""
    # First Plot: Grid Animation
    fig1, ax1 = plt.subplots()
    ax1.set_title('Forest Fire Simulation')

    cmap = ListedColormap([
        'white',  # EMPTY
        '#008000',  # TREE (green)
        '#FF0000',  # ON_FIRE (red)
        '#FF4500',  # BURNED (orange-red)
        '#808080', # PREVIOUSLY_BURNED (grey)
        '#0000FF',  # FIREBREAK (blue)
        '#ADD8E6',  # REMOVED (light blue)
        '#800080',  # SUPPRESSED (purple)
        '#FFC0CB'  # PREVIOUSLY_SUPPRESSED (light pink)
    ])

    # calculate the ticks for the colorbar using an arange between 0 and 1
    tix = 0.05 + np.arange(0, 1, 1 / len(GridState))

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax1, ticks=tix, orientation='vertical', pad=0.2)
    cbar.set_label('Grid States')
    cbar.set_ticklabels([state.name for state in GridState])

    def update(frame):
        ax1.clear()
        ax1.set_title(f'Forest Fire Simulation (Time Step: {frame+1})')
        im = ax1.imshow(grids[frame], cmap=cmap, vmin=0, vmax=8)
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
    
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.title('Wildfire Spread Process Example')
    fig2.tight_layout()

    plt.savefig('wildfire_spread_process_example.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_visualization(grid_size: int, initial_population: int, time_steps: int):
    """Run the simulation and visualization."""
    grids, populations, footprints, extinguishment_time = run_simulation(grid_size, initial_population, time_steps)
        
    if extinguishment_time is not None:
        print(f"The fire was extinguished at time step {extinguishment_time}.")
    else:
        print("The fire was not extinguished within the simulation time.")

    visualize_simulation(grids, populations, footprints)

if __name__ == "__main__":
    run_visualization(grid_size=50, initial_population=10, time_steps=100)