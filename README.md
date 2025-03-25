# RL_Grid

The **RL_Grid** project provides a grid world environment for analyzing maximization bias/ overestimation bias in Reinforcement Learning (RL). A common antidote to maximization bias is Double-Q-Learning. The present code includes Q-Learning and Double-Q-Learning in order to demonstrate the beneficial effect of learning multiple Q-functions. The grid world example 

![RL_Grid_Max](plots/Maximization.png)

*Figure 1: The 2D grid world (left) and the learning curve of Q-Learning with the maximization bias of the non-optimal actions at the beginning of RL training.*

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
**RL_Grid** is written in Python and includes a simple grid world environment and two common RL algorithms. This section details the setup and the issue of maximization bias.

### Maximization bias/ Overestimation bias
Maximization bias is a common issue in RL which can slow down learning considerably. It originates from taking the maximum of a noisy value estimate [1]. For example, the bias may arise by the *argmax* operator in (ε-)greedy action selection and/or the *max* in Q-Learning updates [2]. These maximization operators tend to introduce a significant positive bias in the state-action value q<sub>π</sub>, which can deteriorate the agent performance [3]. --REPHRASE--

### Grid world
To investigate maximization bias, **RL_Grid** applies a simple 2D grid world environment (Fig. 2) with four different states S $\in$ {s<sup>0</sup>, s<sup>1</sup>, s<sup>2</sup>, s<sup>3</sup>}, resembling the finite MDP environment of Sutton and Barto [2] (Fig. 6.5, p. 135). The agent is always initialized in s<sup>0</sup> and can take an action according to its action space A $\in$ {a<sup>n</sup>, a<sup>e</sup>, a<sup>s</sup>, a<sup>w</sup>}. When the agent enters one of the terminal states T, it receives a reward according to the underlying reward model.

![RL_Grid_Grid](plots/GridWorld.png)

*Figure 2: 2D grid world with the optimal path (s<sup>0</sup>, a<sup>e</sup>, s<sup>2</sup>, a<sup>e</sup>, T) [T: Terminal state; p(r|s,a): Probability of the reward r for taking action a in state s; q<sub>π,opt</sub>: State-action value of the optimal path in s<sup>0</sup>; q<sub>π,non-opt</sub>: Maximum state-action value of the non-optimal path in s<sup>0</sup>; α: Learning rate; γ; Discount factor; ε: Exploration coefficient].*

Due to the reward model, the optimal path with the highest expected reward of 1 is (s<sup>0</sup>, a<sup>e</sup>, s<sup>2</sup>, a<sup>e</sup>, T). By contrast, the expected reward for entering T being in s<sup>1</sup>, s<sup>3</sup>, or s<sup>4</sup> only offers 0.76. However, the agent may encounter a reward of 3 with 44 % probability, which can lead to an overestimation of state-action values at the beginning of training. 

### Q-Learning and Double Q-Learning

**Note**: *For the sake of brevity, this subsection outlines only the basic principles for training the Q-Learning and Double Q-Learing algorithms. For more details and comprehensive information, please refer to [2].* 

Q-Learning is a fundamental value-based, off-policy RL algorithm with a temporal difference (TD) update of the state-action value. These q<sub>π</sub> values support the agent in taking beneficial actions in the environment. In a simple but effective manner, the agent can choose the action with the highest q<sub>π</sub> (greedy policy). However, since the agent does not explore the environment in this case, i.e., taking other action than the one with the highest q<sub>π</sub> value, it cannot improve the q<sub>π</sub> estimate of the other actions (for non-optimistic initialization). This can lead to a policy where the agent always takes a non-optimal action. To circumvent this problem, the policy typically includes a probability ε, with which the agent takes a random action (ε-greedy policy). This ensures that all actions are selected sufficiently often on the long run, to build a good estimate of the state-action values.

In Q-Learning, these estimates are derived by a TD update rule. After the agent takes an action a in state s, it observes the reward r and the next state s'. Based on this information, Q-Learning updates q<sub>π</sub>(s,a) using the following update rule:

q<sub>π</sub>(s,a) <- q<sub>π</sub>(s,a) + α ((r + γ max<sub>a'</sub> q<sub>π</sub>(s',a')) - q<sub>π</sub>(s,a))

The update rule adjusts q<sub>π</sub>(s,a) incrementally. The learning rate α determines the size of the update. While low α values result in slow but more stable learning, a high α transfers the last information quite directly to q<sub>π</sub>(s,a) which might lead to instabilities for noisy (s,a,r,s') pairs. On the other hand, the discount factor γ rates the potential future reward - indicated by q<sub>π</sub>(s',a') - to the immediate reward r. 

As presented, Q-Learning with an ε-greedy policy contains two *max* operators, for choosing the greedy policy with probability 1-ε, and for the q<sub>π</sub> update. For this reason, Q-Learning is susceptible to maximization bias. To ameliorate this problem, Hasselt [3] introduced Double Q-Learning. In Double Q-Learning, the agent learns two different Q-function:

q<sub>π,1</sub>(s,a) <- q<sub>π,1</sub>(s,a) + α ((r + γ q<sub>π,2</sub>(s',argmax<sub>a'</sub> q<sub>π,1</sub>(s',a'))) - q<sub>π,1</sub>(s,a))

q<sub>π,2</sub>(s,a) <- q<sub>π,2</sub>(s,a) + α ((r + γ q<sub>π,1</sub>(s',argmax<sub>a'</sub> q<sub>π,2</sub>(s',a'))) - q<sub>π,2</sub>(s,a))

The second Q-function q<sub>π,2</sub> is used for the estimate of the future expected reward in (s', a'). A high bias in q<sub>π,1</sub>(s', a') is therefore not directly transfered to q<sub>π,1</sub>(s,a), but checked by the q<sub>π,2</sub>. This procedure can considerably decrease the maximization bias.

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
- **`src/rl_grid_agents.py`**: Implementation of the Q-Learning and Double Q-Learning algorithms with ε-greedy policies.
  - `QAgent()`: Q-Learning class.
    - `update_policy()`: Updates the policy based on the q<sub>π</sub>.
    - `q_learning_update()`: Updates q<sub>π</sub>(s,a) based on the Q-Learning update rule.
    - `take_action()`: Takes an action according the current ε-greedy policy.
    - `reset()`: Resets the agent.  
  - `DoubleQAgent()`: Double Q-Learning class.
    - `update_policy()`: Updates the policy based on the q<sub>π</sub>.
    - `q_learning_update()`: Updates q<sub>π</sub>(s,a) based on the Double Q-Learning update rule.
    - `take_action()`: Takes an action according the current ε-greedy policy.
    - `reset()`: Resets the agent.  
- **`src/rl_grid_config.py`**: Configuration class for RL_Grid.
- **`src/rl_grid_env.py`**: Contains the grid world environment.
  - `GridWorldEnv()`: Grid world environment class.
    - `_get_obs()`: Returns the current observation.
    - `_get_reward()`: Calculates the reward based on the current state and action.
    - `reset()`: Resets the environment.
    - `step()`: Executes an action in the environment.  
    - `_is_done()`: Checks whether the agent reaches a terminal state and the episode is done.
- **`src/rl_grid_utils.py`**: Provides code for visualization of the training results.
  - `plot_results()`: Generates a plot showing the Q-values averaged over all training runs.

### **Main Script**  
- **`rl_grid_main.py`** – The main script for parallel training the two RL agents in the grid world environment with multiple processes.  
  - `train_agent()` – Function for training a single agent in a separate process.  
  - `main()` – Runs model training and evaluation.  

### **Miscellaneous**  
- **`requirements.txt`** – Lists required Python libraries.

## Installation and Usage
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

## References

[1] S. Thrun, A. Schwartz, "*Issues in using function approximation for reinforcement learning*", Proceedings of the 1993 connectionist models summer school, 1993, 255–263

[2] R. S. Sutton, A. G. Barto, "*Reinforcement Learning: An Introduction*", The MIT Press, Cambridge, Massachusetts, 2018

[3] H. V. Hasselt, "*Double Q-learning*", Advances in neural information processing systems, 23, 2010, 1–9

## Requirements
The project requires the following Python packages:
- numpy
- matplotlib
- tqdm

Make sure to install these packages using the `requirements.txt` file.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.