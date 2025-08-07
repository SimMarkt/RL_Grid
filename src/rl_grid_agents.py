"""
---------------------------------------------------------------------------------------------
RL_Grid: Grid world for analyzing maximization bias in RL using Q-learning and Double Q-learning

rl_grid_agents.py:
> Implements the Q-Learning and Double Q-Learning algorithms for the RL project
---------------------------------------------------------------------------------------------
"""

import numpy as np

from src.rl_grid_config import RLGridConfiguration

class QAgent:
    ''' Implementation of the Q-Learning agent. '''

    def __init__(self, config: RLGridConfiguration, actions: np.ndarray) -> None:
        '''
        Initializes the QAgent.
        :param config: The configuration object for the grid world.
        :param actions: The action space of the environment.
        '''
        self.config = config

        self.actions = actions  # [north, east, south, west]

        # Initialize the policy for every state in the state space.
        # Note that self.policy contains the (greedy) policy learned by Q-Learning.
        self.policy = np.zeros((self.config.row_dim_glob, self.config.column_dim_glob))

        # Initialize state-action values.
        self.q_pi = np.zeros((self.config.row_dim_glob,
                              self.config.column_dim_glob,
                              len(self.actions)))

        # Update the policy.
        self.update_policy()

    def update_policy(self) -> None:
        '''
        Updates the (greedy) policy based on the state-action values.
        '''
        # Loop over the state space.
        for row in range(self.policy.shape[0]):  # Iterate over rows in the grid world state space.
            for col in range(self.policy.shape[1]):  # Iterate over columns.
                # Get the indices of the action-related Q-values sorted in descending order.
                idx_best = np.argpartition(-self.q_pi[row, col, :],
                                           range(self.q_pi[row, col, :].shape[0]))[:]

                # Find the maximum Q-value.
                best_q_value = self.q_pi[row, col, idx_best[0]]

                # Check how many actions have the same maximum Q-value.
                best_indices = [idx for idx in idx_best if self.q_pi[row, col, idx] == best_q_value]

                # Randomly choose one of the best actions.
                self.policy[row, col] = np.random.choice(best_indices)

    def q_learning_update(self, state: np.ndarray, action: int, 
                          reward: float, next_state: np.ndarray) -> None:
        '''
        Updates the state-action value according to the Q-Learning algorithm.
        :param state: The current state of the agent.
        :param action: The action taken by the agent.
        :param reward: The reward received by the agent.
        :param next_state: The next state of the agent.
        '''
        target = reward + self.config.gamma * np.max(self.q_pi[next_state[0], next_state[1], :])
        self.q_pi[state[0], state[1], action] += (
            self.config.alpha * (target - self.q_pi[state[0], state[1], action])
        )

    def take_action(self, state: np.ndarray) -> int:
        '''
        Takes an action according to the current policy, considering epsilon-greedy exploration.
        :param state: The current state of the agent.
        :return action: The action to take.
        '''
        self.update_policy()
        rand = np.random.rand()
        if rand > self.config.epsilon:
            action = int(self.policy[state[0], state[1]])
        else:
            action = np.random.randint(0, 4)
        return action

    def reset(self, config: RLGridConfiguration, actions: np.ndarray) -> None:
        '''
        Resets the agent with a new configuration and action space
        (for more information, refer to __init__()).
        :param config: The new configuration object for the grid world.
        :param actions: The action space of the environment.
        '''
        self.config = config
        self.actions = actions
        self.policy = np.zeros((self.config.row_dim_glob, self.config.column_dim_glob))
        self.q_pi = np.zeros((self.config.row_dim_glob,
                              self.config.column_dim_glob,
                              len(self.actions)))
        self.update_policy()

class DoubleQAgent:
    ''' Implementation of the Double-Q-Learning agent. '''

    def __init__(self, config: RLGridConfiguration, actions: np.ndarray) -> None:
        '''
        Initializes the DoubleQAgent.
        :param config: The configuration object for the grid world.
        :param actions: The action space of the environment.
        '''
        self.config = config

        self.actions = actions  # [north, east, south, west]

        # Initialize the policy for every state in the state space.
        # Note that self.policy contains the (greedy) policy learned by Q-Learning.
        self.policy = np.zeros((self.config.row_dim_glob, self.config.column_dim_glob))

        # Initialize state-action values for the two Q-functions.
        self.q_pi = np.zeros((self.config.row_dim_glob,
                              self.config.column_dim_glob, len(self.actions)))
        self.q_pi_2 = np.zeros((self.config.row_dim_glob,
                                self.config.column_dim_glob, len(self.actions)))
        self.pick = 0

        # Update the policy.
        self.update_policy()

    def argmax(self, q_pi_temp, state) -> int:
        '''
        Helper function to find the action with the maximum Q-value for a specific state.
        :param q_pi_temp: The state-action values.
        :param state: The current state.
        :return action: The action with the maximum Q-value.
        '''
        row, col = state
        # Get the indices of the action-related Q-values sorted in descending order.
        idx_best = np.argpartition(-q_pi_temp[row, col, :],
                                   range(q_pi_temp[row, col, :].shape[0]))[:]

        # Find all indices that share the maximum Q-value.
        best_q_value = q_pi_temp[row, col, idx_best[0]]
        best_indices = [idx for idx in idx_best if q_pi_temp[row, col, idx] == best_q_value]

        # Randomly choose one of the best indices.
        return int(np.random.choice(best_indices))

    def update_policy(self) -> None:
        '''
        Updates the (greedy) policy based on the state-action values.
        '''
        for row in range(self.policy.shape[0]):
            for col in range(self.policy.shape[1]):
                state = np.array([row, col])
                self.policy[row, col] = self.argmax(self.q_pi, state)

    def q_learning_update(self, state: np.ndarray, action: int, 
                          reward: float, next_state: np.ndarray) -> None:
        '''
        Updates the state-action value according to the Double-Q-Learning algorithm.
        :param state: The current state of the agent.
        :param action: The action taken by the agent.
        :param reward: The reward received by the agent.
        :param next_state: The next state of the agent.
        '''
        alpha = self.config.alpha
        gamma = self.config.gamma
        # Update the Q-functions alternately.
        if self.pick % 2:
            target_1 = reward + gamma * self.q_pi_2[next_state[0], next_state[1],
                                                    self.argmax(self.q_pi, next_state)]
            self.q_pi[state[0], state[1], action] += alpha * (
                target_1 - self.q_pi[state[0], state[1], action]
                )
        else:
            target_2 = reward + gamma * self.q_pi[next_state[0], next_state[1],
                                                  self.argmax(self.q_pi_2, next_state)]
            self.q_pi_2[state[0], state[1], action] += alpha * (
                target_2 - self.q_pi_2[state[0], state[1], action]
                )
        self.pick += 1

    def take_action(self, state: np.ndarray) -> int:
        '''
        Takes an action according to the current policy, considering epsilon-greedy exploration.
        :param state: The current state of the agent.
        :return action: The action to take.
        '''
        self.update_policy()
        rand = np.random.rand()
        if rand > self.config.epsilon:
            action = int(self.policy[state[0], state[1]])
        else:
            action = np.random.randint(0, 4)
        return action

    def reset(self, config: RLGridConfiguration, actions: np.ndarray) -> None:
        '''
        Resets the agent with a new configuration and action space
        (for more information, refer to __init__()).
        :param config: The new configuration object for the grid world.
        :param actions: The new action space of the environment.
        '''
        self.config = config
        self.actions = actions
        self.policy = np.zeros((self.config.row_dim_glob, self.config.column_dim_glob))
        self.q_pi = np.zeros((self.config.row_dim_glob,
                              self.config.column_dim_glob, len(self.actions)))
        self.q_pi_2 = np.zeros((self.config.row_dim_glob,
                                self.config.column_dim_glob, len(self.actions)))
        self.pick = 0
        self.update_policy()
