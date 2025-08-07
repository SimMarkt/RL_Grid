"""
---------------------------------------------------------------------------------------------
RL_Grid: Grid world for analyzing maximization bias in RL using Q-learning and Double Q-learning

rl_grid_utils.py:
> Utility functions for visualizing RL training results
---------------------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt

def plot_results(results) -> None:
    '''
    Generates a plot showing the Q-values averaged over all runs as a 
    function of training progress (number of episodes).
    :param results: Dictionary containing training results
    '''

    # Configure plot appearance
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams['axes.linewidth'] = 1

    # Create subplots for Q-Learning and Double Q-Learning
    _, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True, sharey=False)

    # Plot Q-Learning results
    axs[0].grid()
    # axs[0].set_ylim([0, 0.8])  # Uncomment to set y-axis limits
    axs[0].plot(results["Q-Learning"][0], 'gray', linestyle="dotted",
                linewidth=1.7, label='Non-optimal')
    axs[0].plot(results["Q-Learning"][1], 'darkblue',
                linewidth=1.7, label='Optimal')
    axs[0].set_ylabel(r'State-action value q$_{\pi}$(s,a)')

    # Plot Double Q-Learning results
    axs[1].grid()
    # axs[1].set_ylim([0, 0.8])  # Uncomment to set y-axis limits
    axs[1].plot(results["DoubleQ-Learning"][0], 'gray', linestyle="dotted",
                linewidth=1.7, label='Non-optimal')
    axs[1].plot(results["DoubleQ-Learning"][1], 'darkblue',
                linewidth=1.7, label='Optimal')
    axs[1].set_ylabel(r'State-action value q$_{\pi}$(s,a)')
    axs[1].set_xlabel('Episodes')

    # Add labels for the subplots
    textstr = 'Q-Learning'
    props = dict(boxstyle='Square', facecolor='white', alpha=0)
    axs[0].text(0.66, 1.05, textstr, fontsize=14, bbox=props,
                transform=axs[0].transAxes, verticalalignment='center')

    textstr = 'DoubleQ-Learning'
    props = dict(boxstyle='Square', facecolor='white', alpha=0)
    axs[1].text(0.5, 1.05, textstr, fontsize=14, bbox=props,
                transform=axs[1].transAxes, verticalalignment='center')

    # Adjust subplot positions
    box = axs[0].get_position()
    axs[0].set_position([box.x0+0.05, box.y0, box.width, box.height])

    box = axs[1].get_position()
    axs[1].set_position([box.x0+0.05, box.y0, box.width, box.height])

    # Add a legend to the plot
    legend = plt.legend(bbox_to_anchor=(0.04, 2.07, 0.55, 1), loc=3,
                        ncol=1, mode="expand", borderaxespad=0., framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_boxstyle('Square', pad=0.2)

    plt.savefig('plots/Q-Learning_Gridworld_plot.png')

    plt.close()
