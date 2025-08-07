"""
---------------------------------------------------------------------------------------------
RL_Grid: Grid world for analyzing maximization bias in RL using Q-learning and Double Q-learning

rl_grid_env.py:
> Implements the grid world environment for the RL project
---------------------------------------------------------------------------------------------
"""

from typing import Any

import numpy as np

from src.rl_grid_config import RLGridConfiguration

class GridWorldEnv:
    ''' Grid world Environment that follows the Gymnasium interface. '''

    def __init__(self, config: RLGridConfiguration) -> None:
        '''
        Initializes the GridWorldEnv.
        :param config: The configuration object for the grid world.
        '''

        self.config = config
        # 2D state space
        self.terminal = 2   # Represents a terminal state
        self.wall = 1       # Represents a wall
        # Implementation of the 2D state space (x, y)
        self.state_space = np.array(
            [
                [0, self.terminal, self.terminal, self.terminal, 0],
                [self.terminal, 0, 0, 0, 0],
                [self.terminal, 0, 0, 0, self.terminal],
                [self.terminal, 0, 0, 0, 0],
                [0, self.terminal, self.terminal, self.terminal, 0],
            ]
        )
        # Initialize the agent's location
        self.init_state = np.array([2, 2])              # Initial state
        self.state = self.init_state
        self.previous_state = self.state                # Previous state

        # Define the action space
        self.actions = np.array([0, 1, 2, 3])           # [north, east, south, west]
        self.action = self.actions[0]                   # Current action

        # Define the reward, reward probability, and negative reward mappings
        self.reward = None   # Current reward
        self.reward_pos = np.zeros((self.config.row_dim_glob,
                                    self.config.column_dim_glob, 4))    # Positive reward map
        self.reward_neg = np.zeros((self.config.row_dim_glob,
                                    self.config.column_dim_glob, 4))    # Negative reward map
        self.reward_prob = np.ones((self.config.row_dim_glob,
                                    self.config.column_dim_glob, 4))    # Reward probability map

        # Define positions for each type of reward (x, y, action)
        reward_positions = [
            (1, 2, 0), (1, 2, 1), (1, 2, 3),
            (3, 2, 1), (3, 2, 2), (3, 2, 3),
            (2, 1, 0), (2, 1, 2), (2, 1, 3)
        ]

        # Set the reward values and probabilities
        self.set_reward_values(self.reward_pos, reward_positions, self.config.rew_pos)
        self.set_reward_values(self.reward_neg, reward_positions, self.config.rew_neg)
        self.set_reward_values(self.reward_prob, reward_positions, self.config.prob_pos)
        # Positive reward at the end of the optimal path
        self.reward_pos[2, 3, 1] = self.config.rew_pos_opt

        self.k = 0      # Training step counter

    @staticmethod
    def set_reward_values(reward_matrix: np.ndarray,
                          positions: list[tuple[int, int, int]],
                          reward_value: int | float) -> None:
        '''
        Helper function to set values for reward, reward_prob, or reward_neg matrices.
        :param reward_matrix: The matrix (reward, reward_prob, or reward_neg) to update.
        :param positions: A list of tuples where each tuple is (row, column, action).
        :param reward_value: The value to assign to the specified positions.
        '''
        for (row, column, action) in positions:
            reward_matrix[row, column, action] = reward_value

    def _get_obs(self) -> dict[str, Any]:
        '''
        Returns the current observation.
        :return step, state, action, reward in a dictionary
        '''
        return {'step': self.k, 'state': self.state, 'action': self.action, 'reward': self.reward}

    def _get_reward(self) -> int:
        '''
        Calculates the reward based on the current state and action.
        :return reward
        '''

        rand = np.random.rand()
        # Reward calculation considering reward probability:
        if rand <= self.reward_prob[self.previous_state[0], self.previous_state[1], self.action]:
            self.reward = self.reward_pos[self.previous_state[0],
                                          self.previous_state[1], self.action]
        else:
            self.reward = self.reward_neg[self.previous_state[0],
                                          self.previous_state[1], self.action]

        return self.reward

    def reset(self, config: RLGridConfiguration) -> dict[str, Any]:
        '''
        Resets the environment to its initial state.
        :param config: The configuration object for the grid world.
        :return: Initial observation as a numpy array.
        '''
        self.config = config
        self.state = self.init_state
        self.previous_state = self.state
        self.action = self.actions[0]
        self.k = 0

        return self._get_obs()

    def step(self, action) -> tuple[dict[str, Any], int, bool]:
        '''
        Executes an action in the environment.
        :param action: The action to perform.
        :return: The observation, reward, and whether the episode is done.
        '''

        self.action = action
        rand = np.random.rand()
        self.previous_state = np.array([self.state[0], self.state[1]])

        # Transition to the next state with probability dyn_prob
        if rand <= self.config.dyn_prob:
            if action == self.actions[0]:  # Move north
                if self.state[0] == 2 and self.state[1] == 2:       # s0
                    self.state = np.array([1, 2])
                elif self.state[0] == 1 and self.state[1] == 2:     # s1
                    self.state = np.array([0, 2])
                elif self.state[0] == 2 and self.state[1] == 3:     # s2
                    self.state = np.array([2, 3])
                elif self.state[0] == 3 and self.state[1] == 2:     # s3
                    self.state = np.array([2, 2])
                elif self.state[0] == 2 and self.state[1] == 1:     # s4
                    self.state = np.array([1, 0])
                else:
                    raise ValueError(f'Invalid state {self.state} for '+
                                     'action North or Terminal state!')
            elif action == self.actions[1]:  # Move east
                if self.state[0] == 2 and self.state[1] == 2:       # s0
                    self.state = np.array([2, 3])
                elif self.state[0] == 1 and self.state[1] == 2:     # s1
                    self.state = np.array([0, 3])
                elif self.state[0] == 2 and self.state[1] == 3:     # s2
                    self.state = np.array([2, 4])
                elif self.state[0] == 3 and self.state[1] == 2:     # s3
                    self.state = np.array([4, 3])
                elif self.state[0] == 2 and self.state[1] == 1:     # s4
                    self.state = np.array([2, 2])
                else:
                    raise ValueError(f'Invalid state {self.state} for '+
                                     'action East or Terminal state!')
            elif action == self.actions[2]:  # Move south
                if self.state[0] == 2 and self.state[1] == 2:       # s0
                    self.state = np.array([3, 2])
                elif self.state[0] == 1 and self.state[1] == 2:     # s1
                    self.state = np.array([2, 2])
                elif self.state[0] == 2 and self.state[1] == 3:     # s2
                    self.state = np.array([2, 3])
                elif self.state[0] == 3 and self.state[1] == 2:     # s3
                    self.state = np.array([4, 2])
                elif self.state[0] == 2 and self.state[1] == 1:     # s4
                    self.state = np.array([3, 0])
                else:
                    raise ValueError(f'Invalid state {self.state} for '+
                                     'action South or Terminal state!')
            elif action == self.actions[3]:  # Move west
                if self.state[0] == 2 and self.state[1] == 2:       # s0
                    self.state = np.array([2, 1])
                elif self.state[0] == 1 and self.state[1] == 2:     # s1
                    self.state = np.array([0, 1])
                elif self.state[0] == 2 and self.state[1] == 3:     # s2
                    self.state = np.array([2, 2])
                elif self.state[0] == 3 and self.state[1] == 2:     # s3
                    self.state = np.array([4, 1])
                elif self.state[0] == 2 and self.state[1] == 1:     # s4
                    self.state = np.array([2, 0])
                else:
                    raise ValueError(f'Invalid state {self.state} for '+
                                     'action West or Terminal state!')
            else:
                raise ValueError(f'Invalid action {action}. Must be one '+
                                     'of [0, 1, 2, 3] = [north, east, south, west]')

        # Get the next observation
        observation = self._get_obs()
        # Get the reward
        reward = self._get_reward()
        # Check if the episode is done
        done = self._is_done()

        self.k += 1
        return observation, reward, done

    def _is_done(self) -> bool:
        '''
        Checks whether the episode is done.
        :return: True if the current state is a terminal state, False otherwise.
        '''
        if self.state_space[self.state[0], self.state[1]] == self.terminal:
            return True
        else:
            return False
