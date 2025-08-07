"""
---------------------------------------------------------------------------------------------
RL_Grid: Grid world for analyzing maximization bias in RL using Q-learning and Double Q-learning

rl_grid_main.py:
> Main script for running the RL project with multiprocessing
---------------------------------------------------------------------------------------------
"""
import multiprocessing as mp
from typing import Union

import numpy as np

from tqdm import tqdm
from src.rl_grid_config import RLGridConfiguration
from src.rl_grid_env import GridWorldEnv
from src.rl_grid_agents import QAgent, DoubleQAgent
from src.rl_grid_utils import plot_results

def train_agent(agent_class: Union[QAgent,DoubleQAgent],
                config: RLGridConfiguration,
                queue: mp.Queue,
                agent_type: str) -> None:
    """ Train a single agent in a separate process """
    env = GridWorldEnv(config)
    agent = agent_class(config, env.actions)

    # Arrays to store
    # q_max_hist: the maximum Q-value of non-optimal actions
    # q_opt_hist: the Q-value of the optimal action
    q_max_hist = np.zeros((config.num_runs, config.num_episodes))
    q_opt_hist = np.zeros((config.num_runs, config.num_episodes))

    for run in tqdm(range(config.num_runs), desc=f"Training Runs ({agent_type})"):
        agent.reset(config, env.actions)
        for i in range(config.num_episodes):
            state = env.reset(config)['state']
            done = False

            while not done:
                # Select an action based on the current policy
                action = agent.take_action(state)
                # Receive observation and reward from the environment
                obs, reward, done = env.step(action)
                next_state = obs['state']
                # Perform (Double) Q-learning update
                agent.q_learning_update(state, action, reward, next_state)
                state = next_state

            # Record the maximum Q-value of non-optimal actions and
            # the Q-value of the optimal action
            q_max_hist[run, i] = max(
                agent.q_pi[env.init_state[0], env.init_state[1], a]
                for a in [0, 2, 3]
            )
            q_opt_hist[run, i] = agent.q_pi[env.init_state[0], env.init_state[1], 1]

    # Store the averaged Q-values over all runs in the queue
    queue.put((agent_type, np.mean(q_max_hist, axis=0), np.mean(q_opt_hist, axis=0)))

def main() -> None:
    """
    Main function to set up the multiprocessing environment and train agents.
    """
    config = RLGridConfiguration()
    queue = mp.Queue()  # Queue to collect results from processes

    agents = [
        (QAgent, "Q-Learning"),
        (DoubleQAgent, "DoubleQ-Learning")
    ]

    processes = []
    for agent_class, agent_type in agents:
        # Retrieve results from the queue
        p = mp.Process(target=train_agent, args=(agent_class, config, queue, agent_type))
        processes.append(p)
        p.start()

    results = {}
    for _ in agents:
        agent_type, q_max_avg, q_opt_avg = queue.get()
        # q_max_avg: Average maximum Q-value of non-optimal actions over all runs
        # q_opt_avg: Average Q-value of the optimal action over all runs
        results[agent_type] = (q_max_avg, q_opt_avg)

    for p in processes:
        p.join()       # Wait for all processes to finish

    # Extract results and generate plot
    plot_results(results)

if __name__ == "__main__":
    main()
