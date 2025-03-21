# ---------------------------------------------------------------------------------------------
# RLGrid: Grid world for analyzing maximization bias in RL with Q-learning and Double Q-learning
#
# rl_agents.py:
# > Implements the Q-Learning and Double Q-Learning Algorithms for the RL project
# ---------------------------------------------------------------------------------------------

import numpy as np

class QAgent:
    ''' Implementation of the Q-Learning agent. '''

    def __init__(self, RLGridConfiguration, actions):
        '''
            Initialize QAgent.
            :param RLGridConfiguration: The configuration object for the grid world.
            :param actions: The action space of the environment.
        '''
        self.RLGridConfiguration = RLGridConfiguration

        self.actions = actions   # [north, east, south, west]

        # Initialize policy for every state in the state space
        # Note that self.policy contains the (greedy) policy learned by Q-Learning
        self.policy = np.zeros((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB))  # learned policy by Q-Learning
        
        # Initialize state-action values
        self.q_pi = np.zeros((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB, len(self.actions)))
        
        # Update policy
        self.update_policy()

    def update_policy(self):
        '''
            Updates the (greedy) policy based on the state-action values.
        '''
        # Loop over the state space
        for row in range(self.policy.shape[0]):         # Iterate over rows in the Gridworld state space
            for col in range(self.policy.shape[1]):     # Iterate over columns in the Gridworld state space
                # Get the indices of the Q-values sorted in descending order
                idx_best = np.argpartition(-self.q_pi[row, col, :], range(self.q_pi[row, col, :].shape[0]))[:]
                
                # Find the maximum Q-value
                best_q_value = self.q_pi[row, col, idx_best[0]]
                
                # Check how many actions have the same best Q-value
                best_indices = [idx for idx in idx_best if self.q_pi[row, col, idx] == best_q_value]
                
                # Randomly choose one of the best actions
                self.policy[row, col] = np.random.choice(best_indices)

    def q_learning_update(self, state, action, reward, next_state):
        '''
            Updates the state-action value according to the Q-Learning algorithm.
            :param state: The current state of the agent.
            :param action: The action taken by the agent.
            :param reward: The reward received by the agent.
            :param next_state: The next state of the agent.
        '''
        target = reward + self.RLGridConfiguration.gamma * np.max(self.q_pi[next_state[0], next_state[1], :])
        self.q_pi[state[0], state[1], action] += self.RLGridConfiguration.alpha * (target - self.q_pi[state[0], state[1], action])

    def take_action(self, state):
        '''
            Takes an action according to the current policy considering epsilon-greedy exploration.
            :param state: The current state of the agent.
            :return action: The action to take.
        '''
        self.update_policy()
        rand = np.random.rand()
        if rand > self.RLGridConfiguration.epsilon:
            action = int(self.policy[state[0], state[1]])
        else:
            action = np.random.randint(0, 4)
        return action

    def reset(self):
        self.__init__()

class DoubleQAgent:
    ''' Implementation of the Double-Q-Learning agent. '''

    def __init__(self, RLGridConfiguration, actions):
        '''
            Initialize DoubleQAgent.
            :param RLGridConfiguration: The configuration object for the grid world.
            :param actions: The action space of the environment.
        '''
        self.RLGridConfiguration = RLGridConfiguration

        self.actions = actions   # [north, east, south, west]

        # Initialize policy for every state in the state space
        # Note that self.policy contains the (greedy) policy learned by Q-Learning
        self.policy = np.zeros((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB))  # learned policy by Q-Learning
        
        # Initialize state-action values of the two Q-functions
        self.q_pi = np.zeros((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB, len(self.actions)))
        self.q_pi_2 = np.zeros((self.RLGridConfiguration.ROW_DIM_GLOB, self.RLGridConfiguration.COLUMN_DIM_GLOB, len(self.actions)))
        self.pick = 0

        # Update policy
        self.update_policy()

    def argmax(self, q_pi_temp, state):
        '''
            Helper function to find the action with the maximum Q-value of a specifc state.
            :param q_pi_temp: The state-action values.
            :param state: The current state.
            :return action: The action with the maximum Q-value.
        '''
        row, col = state
        # Get the indices of the Q-values sorted in descending order
        idx_best = np.argpartition(-q_pi_temp[row, col, :], range(q_pi_temp[row, col, :].shape[0]))[:]

        # Find all indices that share the maximum Q-value
        best_q_value = q_pi_temp[row, col, idx_best[0]]
        best_indices = [idx for idx in idx_best if q_pi_temp[row, col, idx] == best_q_value]

        # Randomly choose one of the best indices
        return int(np.random.choice(best_indices))

    def update_policy(self):
        '''
            Updates the (greedy) policy based on the state-action values.
        '''
        for row in range(self.policy.shape[0]):
            for col in range(self.policy.shape[1]):
                state = np.array([row, col])
                self.policy[row, col] = self.argmax(self.q_pi, state)

    def q_learning_update(self, state, action, reward, next_state):
        '''
            Updates the state-action value according to the Double-Q-Learning algorithm.
            :param state: The current state of the agent.
            :param action: The action taken by the agent.
            :param reward: The reward received by the agent.
            :param next_state: The next state of the agent.
        '''
        alpha = self.RLGridConfiguration.alpha
        gamma = self.RLGridConfiguration.gamma
        # Update the Q-functions alternately
        if self.pick % 2:
            target_1 = reward + gamma * self.q_pi_2[next_state[0], next_state[1], self.argmax(self.q_pi, next_state)]
            self.q_pi[state[0], state[1], action] += alpha * (target_1 - self.q_pi[state[0], state[1], action])
        else:
            target_2 = reward + gamma * self.q_pi[next_state[0], next_state[1], self.argmax(self.q_pi_2, next_state)]
            self.q_pi_2[state[0], state[1], action] += alpha * (target_2 - self.q_pi_2[state[0], state[1], action])
        self.pick += 1

    def take_action(self, state):
        '''
            Takes an action according to the current policy considering epsilon-greedy exploration.
            :param state: The current state of the agent.
            :return action: The action to take.
        '''
        self.update_policy()
        rand = np.random.rand()
        if rand > self.epsilon:
            action = int(self.policy[state[0], state[1]])
        else:
            action = np.random.randint(0, 4)
        return action

    def reset(self):
        self.__init__()