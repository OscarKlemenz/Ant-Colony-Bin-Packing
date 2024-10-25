""" Script for running the different aco experiments on the 1DBPP
"""
import matplotlib.pyplot as plt
import numpy as np
from packing_graph import PackingGraph
from ant import Ant
import config as conf
import random

def runExperiment(num_ants, evaporation_rate, bins, items, random_seed):
    """ Runs a set amount of evaluations of the ACO algorithm on the 1DBPP

    Args:
        p (int): The number of paths to generate before updating the pheromone values
        e (float): Evaporation rate of the pheromones
        bins (int): The number of bins the items can be placed in
        items (int[]): The items to place in the bins
        random_seed (int): The seed value for when random is used in the experiment
    """
    # Sets the seed
    no_of_evaluations = 0
    best_fitness = float('inf')

    random.seed(99)
    # Generate a graph
    graph = PackingGraph(bins, items) 
    graph.initialiseGraph()

    random.seed(random_seed)

    current_ants = [Ant(graph) for _ in range(num_ants)]

    # Holds best and worst fitness for each p paths
    fitness_prog = []

    while no_of_evaluations < conf.NUM_EVALUATIONS:
        for ant in current_ants:
            ant.traverseGraph()
            current_fitness = ant.getFitness()
            
            # Check and store the best fitness
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_ant = ant
            
            no_of_evaluations += 1

        # Store the best fitness for the paths evaluated
        fitness_prog.append(best_fitness)

        # Batch pheromone updates
        for ant in current_ants:
            ant.updatePathPheromones()
        
        # Evaporate the pheromones after all updates
        graph.evaporatePheromones(evaporation_rate)

    
    # Calculate the average difference between all the different bins
    sorted_keys = sorted(best_ant.getBinWeights().keys())
    sorted_values = [best_ant.getBinWeights()[key] for key in sorted_keys]
    differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values) - 1)]
    average_difference = sum(differences) / len(differences)
    
    # Outputs
    print("Best fitness: ", best_fitness)
    print("Average difference of bins for best fitness: ", abs(average_difference))

    return fitness_prog, best_fitness

def getValuesForProblem(bin_problem):
    """ Returns the correct bin and item values for a set problem

    Args:
        bin_problem (str): The type of bin problem to undertake

    Returns:
        int, int[]: The number of bins and the items to be placed in the bins
    """
    items = []
    if bin_problem == "BPP1":
        items = list(range(1, 501))
        return 10, items
    elif bin_problem == "BPP2":
        items = [i ** 2 / 2 for i in range(1, 501)]
        return 50, items
    else:
        return 0, items

def plotBestFitnessProgression(best_fitnesses):
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

def plotExperimentTrials(data):
    """
    Plots a grouped bar chart for multiple experiments, each with multiple trials.
    
    Parameters:
    - data (int[][]): List of lists, where each inner list contains fitness values for each trial in an experiment.
    """
    num_experiments = len(data)
    num_trials = len(data[0]) if num_experiments > 0 else 0
    
    experiment_labels = [f"Experiment {i+1}" for i in range(num_experiments)]
    
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

if __name__ == "__main__":
    
    # Gets the bin num and items for the set problem
    bins, items = getValuesForProblem(conf.BPP1)
    
    plot_fitnesses_progression = True
    plot_experiment_trials = True
    
    all_fitnesses = []

    # For each of the experiments
    for values in conf.P_AND_E_VALUES:
        experiment_fitnesses = []
        fitness_progressions = []
        # Runs a set amount of trials
        for i in range(0, conf.NUM_TRIALS):
            # i is used as the random seed for each trial
            fitness_prog, best_fitness = runExperiment(values[0], values[1], bins, items, i)

            fitness_progressions.append(fitness_prog)
            experiment_fitnesses.append(best_fitness)
            # Could do range / sum as well
        
        all_fitnesses.append(experiment_fitnesses)

        if plot_fitnesses_progression: plotBestFitnessProgression(fitness_progressions)
    
    if plot_experiment_trials: plotExperimentTrials(all_fitnesses)
    

        