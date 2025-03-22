# ---------------------------------------------------------------------------------------------
# RL_Grid: Grid world for analyzing maximization bias in RL with Q-learning and Double Q-learning
#
# rl_grid_main.py:
# > Main script for running the RL project
# ---------------------------------------------------------------------------------------------

import numpy as np
from tqdm import tqdm
from src.rl_grid_config import RLGridConfiguration
from src.rl_grid_env import GridWorldEnv
from src.rl_grid_agents import QAgent, DoubleQAgent
from src.rl_grid_utils import plot_results

def main():
    # Load the configuration, the environment, and the agents
    RLGridConfig = RLGridConfiguration()
    env = GridWorldEnv(RLGridConfig)
    agent_Q = QAgent(RLGridConfig, env.actions)
    agent_DQ = DoubleQAgent(RLGridConfig, env.actions)

    # Arrays storing the Q-values for the entire history of training runs and episodes
    q_init_max_not_ae_history = np.zeros((RLGridConfig.num_runs, RLGridConfig.num_episodes))        # The maximum Q-Value of all non-optimal actions
    q_init_ae_history = np.zeros((RLGridConfig.num_runs, RLGridConfig.num_episodes))                # Q-Value of the optimal action  

    # Arrays storing the Q-values in average over the number of runs for the two different Agents
    q_max_not_ae_avg = np.zeros((RLGridConfig.num_episodes, 2))             # The maximum Q-Value of all non-optimal actions
    q_ae_avg = np.zeros((RLGridConfig.num_episodes, 2))                     # Q-Value of the optimal action

    for agent in [agent_Q, agent_DQ]:  # Loop over the two agents
        
        for run in tqdm(range(RLGridConfig.num_runs), desc="Training Runs ("):
            agent.reset(RLGridConfig, env.actions)
            for i in range(RLGridConfig.num_episodes):
                obs = env.reset(RLGridConfig)
                state = obs[1:3]
                done = False

                while not done:
                    action = agent.take_action(state)
                    obs, reward, done = env.step(action)
                    next_state = obs[1:3]

                    agent.q_learning_update(state, action, reward, next_state)
                    state = next_state

                q_init_max_not_ae_history[run, i] = max(agent.q_pi[2, 2, 0], agent.q_pi[2, 2, 2], agent.q_pi[2, 2, 3])
                q_init_ae_history[run, i] = agent.q_pi[2, 2, 1]

        q_init_max_not_ae_avg = np.average(q_init_max_not_ae_history, axis=0)
        q_ae_avg = np.average(q_init_ae_history, axis=0)

        q_max_not_ae_avg[:, a] = q_init_max_not_ae_avg
        q_ae_avg[:, a] = q_ae_avg

    plot_results(q_max_not_ae_avg, q_ae_avg)

if __name__ == "__main__":
    main()