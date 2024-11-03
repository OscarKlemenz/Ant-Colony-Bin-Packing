import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

def plotExperimentTrials(data):
    """
    Plots a grouped bar chart for multiple experiments, each with multiple trials.
    
    Parameters:
    - data (int[][]): List of lists, where each inner list contains fitness values for each trial in an experiment.
    """
    num_experiments = len(data)
    num_trials = len(data[0]) if num_experiments > 0 else 0
    
    experiment_labels = ["100, 0.9", "100, 0.6", "10, 0.9", "10, 0.6"]
    
    # Plotting parameters
    bar_width = 0.15
    space_between_experiments = 0.2
    
    # X positions for each group
    x_positions = np.arange(num_experiments) * (num_trials * bar_width + space_between_experiments)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting each trial within each experiment
    for i in range(num_trials):
        trial_positions = x_positions + i * bar_width
        trial_data = [experiment[i] for experiment in data]  # Extract each trial's data across experiments
        ax.bar(trial_positions, trial_data, width=bar_width, label=f'Trial {i+1}', zorder=3)
    
    # Customizing the plot
    ax.set_xticks(x_positions + (num_trials - 1) * bar_width / 2)
    ax.set_xticklabels(experiment_labels)
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Results for Each Experiment and Trial")
    # Format the y-axis with commas for thousands using a lambda function
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    #ax.set_yscale('log')
    ax.set_ylim(300000, 600000)  # Set y-axis limits here
    ax.legend(title="Trials")
    ax.grid(which='both', linestyle='--', linewidth=0.7, zorder=1)
    plt.show()

data = [[440679,483734.5,472905.0,442361.5,464977.0],
        [441116.0,436472.0,456628.0,479934.0,450805.0],
        [449646.5,431968.5,473701.5,447405.5,418265.0],
        [507444.5,487475.5,471667.5,425316.5,528085.5]]
plotExperimentTrials(data)