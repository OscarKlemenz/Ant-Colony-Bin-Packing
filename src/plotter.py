""" Data visualisation functions
"""
import matplotlib.pyplot as plt

def plotBestFitnessProgressionAllTrials(best_fitnesses):
    """ Line graph of how the best fitness evolves over each of the trials for an experiment

    Args:
        best_fitnesses (int[][]): List of the best fitnesses for each trial
    """
    # Use the indices of the arrays as the test numbers
    test_numbers = range(len(best_fitnesses[0]))  # Assuming all trials have the same number of tests

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Loop over each trial in best_fitnesses and plot them
    for idx, trial_best_values in enumerate(best_fitnesses):
        plt.plot(test_numbers, trial_best_values, linestyle='-', label=f'Trial {idx + 1}')

    # Add title and labels
    plt.title('Best Fitness')
    plt.xlabel('Test Number')
    plt.ylabel('Fitness')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()

def plotBestFitnessProgressionOneTrial(best_fitnesses):
    """ Line graph of how the best fitness evolves over a trial

    Args:
        best_fitnesses (list of int): List of the best fitnesses for a single trial
    """
    # Use the indices of the array as the test numbers
    test_numbers = range(len(best_fitnesses))  # Length of the best_fitnesses array

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the best fitness values for the single trial
    plt.plot(test_numbers, best_fitnesses, linestyle='-', label='Trial 1')

    # Add title and labels
    plt.title('Best Fitness Progression')
    plt.xlabel('Test Number')
    plt.ylabel('Fitness')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()

def plotExperimentTrials(data):
    """
    Plots a grouped bar chart for multiple experiments, each with multiple trials.
    
    Parameters:
    - data (int[][]): List of lists, where each inner list contains fitness values for each trial in an experiment.
    """
    num_experiments = len(data)
    num_trials = len(data[0]) if num_experiments > 0 else 0
    
    #experiment_labels = [f"Experiment {i+1}" for i in range(num_experiments)]
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
        ax.bar(trial_positions, trial_data, width=bar_width, label=f'Trial {i+1}')
    
    # Customizing the plot
    ax.set_xticks(x_positions + (num_trials - 1) * bar_width / 2)
    ax.set_xticklabels(experiment_labels)
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Results for Each Experiment and Trial")
    ax.legend(title="Trials")
    plt.show()

def plotWeightDistribution(weights):
    """
    Plot a bar graph for the distribution of weights.

    Parameters:
    weights (int[]): An array of weights for each bin.
    """
    
    # Create an array of indices for the x-axis
    indices = list(range(len(weights)))

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(indices, weights, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Bin Number')
    plt.ylabel('Weight')
    plt.title('Weight Distribution of Bins')
    plt.xticks(indices)  # Set x-ticks to be the indices of weights
    plt.grid(axis='y')

    # Show the plot
    plt.show()
