# RL_Grid

The **RL_Grid** project provides a grid world environment for analyzing maximization bias/ overestimation bias in Reinforcement Learning (RL). A common antidote to maximization bias is Double-Q-Learning. The present code includes Q-Learning and Double-Q-Learning in order to demonstrate the beneficial effect of learning multiple Q-functions. The grid world example 

![RL_PtG_int](plots/GridWorld.png)

*Figure 1: 2D grid world with the optimal path (s<sup>0</sup>, a<sup>e</sup>, s<sup>2</sup>, a<sup>e</sup>, T) [T: Terminal state; p(r|s,a): Probability of the reward r for taking action a in state s; q<sub>π,opt</sub>: State-action value of the optimal path in s<sup>0</sup>; q<sub>π,non-opt</sub>: Maximum state-action value of the non-optimal path in s<sup>0</sup>; α; : Learning rate; γ; ; Discount factor; ε: Exploration coefficient].*

---

## Table of Contents

1. [Overview](#overview)
2. [Project structure](#project-structure)
3. [Installation and Usage](#installation-and-usage)
4. [License](#license)
5. [Citing](#citing)
6. [References](#references)
7. [Acknowledgments](#acknowledgments)

---


## Overview
**RL_Grid** is written in Python and includes a simple grid world environment and two common RL algorithms. This section details the setup.

### Maximization bias/ Overestimation bias

### Grid world

### Q-Learning and Double Q-Learning

## Project Structure

The project is organized into the following directories and files:

```plaintext
RL_Grid
├── config
│   └── config.py
│
├── src
│   ├── rl_grid_agents.py
│   ├── rl_grid_config.py
│   ├── rl_grid_env.py
│   └── rl_grid_utils.py
│
├── requirements.txt
└── rl_grid_main.py

```

### `config/config.yaml` 
Contains configuration file for the project.  

### `src/`
Contains source code for RL agents and the environment:  
- **`src/rl_config_train.py`**: Preprocesses training settings.
  - `TrainConfiguration()`: Training class.
- **`data/OP.../data-meth_cooldown.csv`**: Cold startup data.

## Installation
To set up the project, clone the repository and install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

## Usage
To run the project, execute the `main.py` file:

```bash
python src/main.py
```

This will initialize the grid world environment, create the reinforcement learning agents, and start the training process. After training, the results will be plotted automatically.

## Components

### Environment
- **GridWorldEnv**: This class implements the grid world environment. It includes methods for:
  - Initializing the environment
  - Stepping through actions
  - Resetting the environment
  - Checking if an episode is done

### Agents
- **QAgent**: Implements the Q-learning algorithm.
- **DoubleQAgent**: Implements the Double Q-learning algorithm.

Both agents have methods for:
- Taking actions
- Updating policies
- Performing Q-learning updates

### Utilities
- **Plotting Functions**: Utility functions for visualizing results, including efficiency plots and reward histories.

## Requirements
The project requires the following Python packages:
- numpy
- matplotlib
- tqdm

Make sure to install these packages using the `requirements.txt` file.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.