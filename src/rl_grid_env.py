# ---------------------------------------------------------------------------------------------
# RL_Grid: Grid world for analyzing maximization bias in RL with Q-learning and Double Q-learning
#
# rl_grid_env.py:
# > Implements the grid world environment for the RL project
# ---------------------------------------------------------------------------------------------

import numpy as np

class GridWorldEnv:
    ''' Gridworld Environment that follows Gymnasium interface. '''
    
    def __init__(self, RLGridConfiguration):
        '''
            Initialize GridWorldEnv.
            :param RLGridConfiguration: The configuration object for the grid world.
        '''

        self.RLGridConfiguration = RLGridConfiguration
        # 2D state space
        self.terminal = 2   # Indicates terminal state
        self.wall = 1       # Indicates wall
        # Implementation of the 2D state space (x,y)
        self.state_space = np.array(
            [
                [0, self.terminal, self.terminal, self.terminal, 0],
                [self.terminal, 0, 0, 0, 0],
                [self.terminal, 0, 0, 0, self.terminal],
                [self.terminal, 0, 0, 0, 0],
                [0, self.terminal, self.terminal, self.terminal, 0],
            ]
        )  
        # Initialize location
        self.state = np.array([2, 2])           # Current (initial) state
        self.previous_state = self.state        # Previous state

        # Define action space
        self.actions = np.array([0, 1, 2, 3])   # [north,east,south,west]
        self.action = self.actions[0]           # Current action

        # Define the reward, reward probability, and negative reward mappings
        self.reward = None                                                                                                  # Current reward
        self.reward_pos = np.zeros((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB, 4))    # Map with positive rewards
        self.reward_neg = np.zeros((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB, 4))    # Map with positive rewards
        self.reward_prob = np.ones((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB, 4))    # Map with reward probabilities

        # Define positions for each type of reward (x,y,action)
        reward_positions_pos = [
            (2, 3, 1), (1, 2, 0), (1, 2, 1), (1, 2, 3),
            (3, 2, 1), (3, 2, 2), (3, 2, 3), (2, 1, 0),
            (2, 1, 2), (2, 1, 3)
        ]
        reward_positions_neg = [
            (1, 2, 0), (1, 2, 1), (1, 2, 3),
            (3, 2, 1), (3, 2, 2), (3, 2, 3),
            (2, 1, 0), (2, 1, 2), (2, 1, 3)
        ]
        reward_prob_positions = reward_positions_neg

        # Set the values using the helper function
        self.set_reward_values(self.reward_pos, reward_positions_pos, self.RLGridConfiguration.rew_pos)
        self.set_reward_values(self.reward_neg, reward_prob_positions, self.RLGridConfiguration.rew_neg)
        self.set_reward_values(self.reward_prob, reward_prob_positions, self.RLGridConfiguration.prob_pos)

        self.k = 0      # Training step

    @staticmethod
    def set_reward_values(reward_matrix, positions, reward_value):
        '''
            Helper function to set the values for reward, reward_prob, or reward_neg matrices.
            :param reward_matrix: The matrix (reward, reward_prob, or reward_neg) to update.
            :param positions: A list of tuples where each tuple is (row, column, action).
            :param reward_value: The value to assign to the specified positions.
        '''
        for (row, column, action) in positions:
            reward_matrix[row, column, action] = reward_value

    def _get_obs(self):
        '''
            Returns the current observations
        '''
        return [self.k, self.state[0], self.state[1], self.action, self.reward]

    def _get_reward(self):
        '''
            Calculates the reward based on the current state and action
        '''
    
        rand = np.random.rand()
        # Reward calculation including reward probability:
        if rand <= self.reward_prob[self.previous_state[0], self.previous_state[1], self.action]:
            self.reward = self.reward_pos[self.previous_state[0], self.previous_state[1], self.action]
        else:
            self.reward = self.reward_neg[self.previous_state[0], self.previous_state[1], self.action]

        return self.reward

    def reset(self, RLGridConfiguration):
        '''
            Important: the observation must be a numpy array
            :return: Observations
        '''

        self.__init__(RLGridConfiguration)
        return self._get_obs() 

    def step(self, action):
        '''
            Performs an action in the environment.
            :param action: The action to perform.
            :return: The observation, reward, and whether the episode is done.
        '''

        self.action = action
        rand = np.random.rand()
        self.previous_state = np.array([self.state[0], self.state[1]])

        # Only perform transition to next state with probability dyn_prob
        if rand <= self.RLGridConfiguration.dyn_prob:
            if action == self.actions[0]:  # go north
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
                    assert False, 'Non valid state or Terminal state!'
            elif action == self.actions[1]:  # go east
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
                    assert False, 'Non valid state or Terminal state!'
            elif action == self.actions[2]:  # go south
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
                    assert False, 'Non valid state or Terminal state!'
            elif action == self.actions[3]:  # go west
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
                    assert False, 'Non valid state or Terminal state!'
            else:
                assert False, 'Action need to match [0, 1, 2, 3] = [north,east,south,west]'

        # Get next observation
        observation = self._get_obs()
        # Get reward
        reward = self._get_reward()
        # Check if episode is done
        done = self._is_done()

        self.k += 1
        return observation, reward, done

    def close(self):
        pass

    def _is_done(self):
        """
        Optional method. Returns whether the episode is done.
        """
        if self.state_space[self.state[0], self.state[1]] == self.terminal:
            return True
        else:
            return False