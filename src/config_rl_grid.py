# ---------------------------------------------------------------------------------------------
# RLGrid: Grid world for analyzing maximization bias in RL with Q-learning and Double Q-learning
#
# config_rl_grid.py:
# > Stores parameters for grid world and the (Double) Q-learning agents
# > Parses the config_rl_grid.yaml data into a class object for further processing
# ---------------------------------------------------------------------------------------------

import yaml

class RLGridConfiguration():
    def __init__(self):
        # Load the environment configuration from the YAML file
        with open("config/config_rl_grid.yaml", "r") as env_file:
            grid_config = yaml.safe_load(env_file)

        # Unpack data from dictionary
        self.__dict__.update(grid_config)

        self.ROW_DIM_GLOB = 5  # grid world dimensions
        self.COLUMN_DIM_GLOB = 5  # grid world dimensions