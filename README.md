# Wildfire Spread Process: A Study of "The birth-death-suppression Markov process and wildfires"

Welcome to this repository! :fire: This repo is dedicated to the exploration and interpretation of the paper "The birth-death-suppression Markov process and wildfires" by George Hulsey, David L. Alderson, and Jean Carlson. 

## What's Inside? :file_folder:

The repository includes my notes and code, inspired by the ideas from the mentioned paper. Here's what each file in this repository serves:

- `./birth_death_example.py`: An illustrative example demonstrating the birth-death process.
- `./eqn16_absorption.png`, `./eqn16_escape.png`, `./eqn16_var_pop_linear.png`: Diagrams visualizing key findings, and equations from the paper.
- `./eqn16_mean_pop.py`, `./eqn16_var_pop.py`: Python files where I've implemented and explored the implications of equation 16 from the paper.
- `./fig2.py`, `./fig3.py`: Code files to generate figures 2 and 3 from the paper.
- `./fig2.png`, `./fig3.png`: The PNG files for figures 2 and 3, recreated from the paper.
- `./notes.pdf`: My raw notes on the paper.
- `./wildfire_model.py`: The heart of the project - the wildfire model. It's a Python file containing the code for simulating a wildfire's spread and control using a birth-death-suppression Markov process.

## The Fire :fire: is in the Details - Diving into `wildfire_model.py`

The `wildfire_model.py` file contains the `WildfireSpreadProcess` class, which encapsulates a continuous-time Markov process simulating the spread of wildfires. This class utilizes various rates: spread rate, extinguish rate, and firefighting rate. It also accommodates the total burnable area size. The rates can either be a constant value or a function of the population that returns the rate, allowing for a versatile wildfire model. 

To provide a taste of how this model behaves, the file also includes a script in the `__main__` section, where a wildfire spread process object is created with rates depending on the current population and a system size of 1000. The process is simulated over 1000 time steps with an initial fire size of 100, and the results (active fire and burned area over time) are plotted. 

## Getting Started :runner:

To dive into this fascinating journey of understanding the wildfire spread process, start by cloning this repository:

```bash
git clone <url-to-this-repo>
cd <repo-directory>
```

Next, ensure you have Python 3.6 or later installed. You can verify this by running:

```bash
python --version
```

This project depends on a couple of Python libraries, `numpy` and `matplotlib`. You can install these via pip:

```bash
pip install numpy matplotlib
```

You are now ready to run the simulations! Navigate to the `wildfire_model.py` file and run:

```bash
python wildfire_model.py
```

This will simulate the wildfire spread process and show a graph illustrating the fire's active size and burned area over time.

To explore some the equations and figures from the paper, simply run the corresponding Python files. For example, to generate figure 2, run:

```bash
python fig2.py
```

## Contributing :handshake:

Contributions are always welcome! Feel free to raise issues, propose enhancements, or even improve the code/documentation via pull requests. Let's spread knowledge, not fire!

## Acknowledgements :clap:

A shoutout to George Hulsey, David L. Alderson, and Jean Carlson for their excellent paper that was the inspiration behind this repository. If you've enjoyed this exploration, I highly recommend you give their paper a read.

