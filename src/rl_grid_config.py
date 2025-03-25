# ---------------------------------------------------------------------------------------------
# RL_Grid: Grid world for analyzing maximization bias in RL using Q-learning and Double Q-learning
#
# rl_grid_config.py:
# > Contains parameters for the grid world environment and the (Double) Q-learning agents.
# > Parses the configuration data from a YAML file into a class object for further use.
# ---------------------------------------------------------------------------------------------

import yaml

class RLGridConfiguration():
    ''' Configuration class for RL_Grid. '''

    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config.yaml", "r") as env_file:
            grid_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(grid_config)

        self.ROW_DIM_GLOB = 5  # Number of rows in the grid world
        self.COLUMN_DIM_GLOB = 5  # Number of columns in the grid world

        self.__print_init()

    def __print_init(self):
        print('\n--------------------------------------------------------------------------------------------')    
        print('-----------------RL_Grid: Grid world for analyzing maximization bias in RL------------------')
        print('--------------------------------------------------------------------------------------------\n')

        print('RL_Grid Configuration:')
        print(f"> Grid Dimensions: \t\t\t\t\t {self.ROW_DIM_GLOB} rows x {self.COLUMN_DIM_GLOB} columns")
        print(f"> Dynamic Probability (dyn_prob): \t\t\t {self.dyn_prob}")
        print(f"> Positive Reward Probability (prob_pos): \t\t {self.prob_pos}")
        print(f"> Positive Reward for Optimal Pathway (rew_pos_opt): \t {self.rew_pos_opt}")
        print(f"> General Positive Reward (rew_pos): \t\t\t {self.rew_pos}")
        print(f"> Negative Reward (rew_neg): \t\t\t\t {self.rew_neg}")
        print(f"> Learning Rate (alpha): \t\t\t\t {self.alpha}")
        print(f"> Discount Factor (gamma): \t\t\t\t {self.gamma}")
        print(f"> Epsilon for Epsilon-Greedy Policy (epsilon): \t\t {self.epsilon}")
        print('--------------------------------------------------------------------------------------------\n')
