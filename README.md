# RL_Project

## Overview
This project implements a reinforcement learning framework using a grid world environment. It includes various reinforcement learning agents and utilities for plotting results.

## Project Structure
```
RL_Project
├── src
│   ├── environments
│   │   └── gridworld_env.py
│   ├── agents
│   │   └── rl_agents.py
│   ├── utils
│   │   └── plotting.py
│   └── main.py
├── requirements.txt
└── README.md
```

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