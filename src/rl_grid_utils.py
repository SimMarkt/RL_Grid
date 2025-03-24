# ---------------------------------------------------------------------------------------------
# RL_Grid: Grid world for analyzing maximization bias in RL with Q-learning and Double Q-learning
#
# rl_grid_utils.py:
# > Utility functions for plotting the RL training results
# ---------------------------------------------------------------------------------------------

def plot_results(results):
    '''
        Creates a plot with the Q-Values averaged over all runs and dependent on the training progress (# of episodes)
        :param results: Dictionary with training results
    '''
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams['axes.linewidth'] = 1

    fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True, sharey=False)
    
    axs[0].grid()
    # axs[0].set_ylim([0, 0.8])
    axs[0].plot(results["Q-Learning"][0], 'gray', linestyle="dotted", linewidth=1.7, label='Non-optimal')
    axs[0].plot(results["Q-Learning"][1], 'darkblue', linewidth=1.7, label='Optimal')
    axs[0].set_ylabel('State-action value q$_{\pi}$(s,a)')

    axs[1].grid()
    # axs[1].set_ylim([0, 0.8])
    axs[1].plot(results["DoubleQ-Learning"][0], 'gray', linestyle="dotted", linewidth=1.7, label='Non-optimal')
    axs[1].plot(results["DoubleQ-Learning"][1], 'darkblue', linewidth=1.7, label='Optimal')
    axs[1].set_ylabel('State-action value q$_{\pi}$(s,a)')
    axs[1].set_xlabel('Episodes')

    textstr = 'Q-Learning'
    props = dict(boxstyle='Square', facecolor='white', alpha=0)
    axs[0].text(0.66, 1.05, textstr, fontsize=14, bbox=props, transform=axs[0].transAxes, verticalalignment='center')

    textstr = 'DoubleQ-Learning'
    props = dict(boxstyle='Square', facecolor='white', alpha=0)
    axs[1].text(0.5, 1.05, textstr, fontsize=14, bbox=props, transform=axs[1].transAxes, verticalalignment='center')

    box = axs[0].get_position()
    axs[0].set_position([box.x0+0.05, box.y0, box.width, box.height])

    box = axs[1].get_position()
    axs[1].set_position([box.x0+0.05, box.y0, box.width, box.height])

    legend = plt.legend(bbox_to_anchor=(0.04, 2.07, 0.55, 1), loc=3, ncol=1, mode="expand", borderaxespad=0., framealpha=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    frame.set_boxstyle('Square', pad=0.2)

    plt.savefig(f'plots/Q-Learning_Gridworld_plot.png')

    plt.close()
