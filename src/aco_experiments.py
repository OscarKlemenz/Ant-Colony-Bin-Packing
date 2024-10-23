""" Script for running the different aco experiments on the 1DBPP
"""
import matplotlib.pyplot as plt
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
    random.seed(random_seed)
    no_of_evaluations = 0
    best_fitness = float('inf')

    # Generate a graph
    graph = PackingGraph(bins, items, random_seed)
    graph.initialiseGraph()

    # Holds best and worst fitness for each p paths
    best_fitnesses = []

    while no_of_evaluations < conf.NUM_EVALUATIONS:
        
        current_ants = []
        # 0-p ants traverse the graph
        for _ in range(0, num_ants):
            # Traverse the graph
            ant = Ant(graph)
            ant.traverseGraph()
            # Store the Ant
            current_ants.append(ant)
            current_fitness = ant.getFitness()
            # Get the fitness and see if its better than current best 
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_ant = ant
            
            no_of_evaluations += 1

        # Store the best and worst for those paths
        best_fitnesses.append(best_fitness)
        # Update the pheromone paths by looping over list of ants and running the method
        for ant in current_ants:
            ant.updatePathPheromones()
        # Evaporate the pheromone
        graph.evaporatePheromones(evaporation_rate)
    
    # AVERAGE CALCULATIONS
    sorted_keys = sorted(best_ant.getBinWeights().keys())
    sorted_values = [best_ant.getBinWeights()[key] for key in sorted_keys]
    # Calculate differences between consecutive values
    differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values) - 1)]
    # Calculate the average of the differences
    average_difference = sum(differences) / len(differences)

    print("Best fitness: ", best_fitness)
    print("Average difference of bins for best fitness: ", abs(average_difference))

    return best_fitnesses

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

if __name__ == "__main__":
    
    # Gets the bin num and items for the set problem
    bins, items = getValuesForProblem(conf.BPP1)
    
    best_fitnesses = []
    
    # Runs a set amount of trials
    for i in range(0, conf.NUM_TRIALS):
        # i is used as the random seed for each trial
        best = runExperiment(10, 0.9, bins, items, i)

        best_fitnesses.append(best)
        # Could do range / sum as well
    
    # Use the indices of the arrays as the test numbers
    test_numbers = range(len(best_fitnesses[0]))  # Assuming all trials have the same number of tests

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Loop over each trial in best_fitnesses and plot them
    for idx, trial_best_values in enumerate(best_fitnesses):
        plt.plot(test_numbers, trial_best_values, linestyle='-', label=f'Best Values - Trial {idx + 1}')

    # Add title and labels
    plt.title('Best vs Worst Test Values Across Multiple Trials')
    plt.xlabel('Test Number')
    plt.ylabel('Value')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()