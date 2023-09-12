# Combined Wildfire Spread Model: Temporal and Spatial Dynamics

Welcome to this repository! :fire: This repository is dedicated to the study and simulation of wildfire spread using both temporal and spatial models. It is inspired by various models that aim to understand the complex nature of wildfires and how they spread over time and space.

## What's Inside? :file_folder:

Here's what each file in this repository serves:

- `./temporal_model.py`: Contains the `WildfireSpreadProcess` class, which models the temporal dynamics of a wildfire spread.
  
- `./spatial_model.py`: Contains classes and functions for the spatial model, including the `Environment`, `System`, and `Optimization` classes.
  
- `./combined_model.py`: Merges the temporal and spatial models into a unified model.
  
- `./visualization.py`: A Python script for visualizing the wildfire spread on a grid over time.

## Diving into the Combined Model :fire:

The `combined_model.py` file contains the `CombinedModel` class, which integrates both temporal and spatial aspects of wildfire spread. This allows the model to be more versatile and realistic, accommodating various conditions and scenarios. 

The temporal model, defined in `temporal_model.py`, follows a continuous-time Markov process, allowing for various rates like spread rate, extinguish rate, and firefighting rate. 

The spatial model, defined in `spatial_model.py`, utilizes a grid to represent the environment and employs stochastic methods to simulate the spread of fire across the grid.

The visualization in `visualization.py` provides a graphical representation of the wildfire spread over time, making it easier to understand the model's behavior.

## Getting Started :runner:

To get started, clone this repository:

```bash
git clone https://github.com/joseph-crowley/combined-wildfire-model.git
cd combined-wildfire-model
```

Ensure you have Python 3.6 or later installed. Verify by running:

```bash
python --version
```

Install required Python libraries:

```bash
pip install numpy matplotlib
```

Run the visualization script to see the model in action:

```bash
python visualization.py
```

## Contributing :handshake:

Contributions are always welcome! Feel free to raise issues, propose enhancements, or improve the code/documentation via pull requests. Let's collaborate to improve our understanding of wildfires!
