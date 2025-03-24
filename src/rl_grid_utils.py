# ---------------------------------------------------------------------------------------------
# RL_Grid: Grid world for analyzing maximization bias in RL with Q-learning and Double Q-learning
#
# rl_grid_utils.py:
# > Utility functions for plotting the RL training results
# ---------------------------------------------------------------------------------------------

def plot_results(results):
    import matplotlib.pyplot as plt



    plt.rcParams.update({'font.size': 14})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams['axes.linewidth'] = 1

    fig, axs = plt.subplots(2, 1, figsize=(4, 5), sharex=True, sharey=False)
    
    axs[0].grid()
    axs[0].set_ylim([0, 0.8])
    axs[0].plot(results["Q-Learning"][0], 'gray', linestyle="dotted", linewidth=1.7, label='Non-optimal')
    axs[0].plot(results["Q-Learning"][1], 'darkblue', linewidth=1.7, label='Optimal')
    axs[0].set_ylabel('State-action value q$_{\pi}$')

    axs[1].grid()
    axs[1].set_ylim([0, 0.8])
    axs[1].plot(results["DoubleQ-Learning"][0], 'gray', linestyle="dotted", linewidth=1.7, label='Non-optimal')
    axs[1].plot(results["DoubleQ-Learning"][1], 'darkblue', linewidth=1.7, label='Optimal')
    axs[1].set_ylabel('State-action value q$_{\pi}$')
    axs[1].set_xlabel('Episodes')

    plt.savefig(f'plots/Q-Learning_Gridworld_plot.png')

    plt.close()

def plot_reward_history(cum_rew_history):
    import matplotlib.pyplot as plt

    plt.plot(cum_rew_history)
    plt.title('Cumulative Reward History')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.grid()
    plt.show()