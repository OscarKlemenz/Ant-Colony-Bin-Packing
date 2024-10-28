""" Script for running the different aco experiments on the 1DBPP
"""
import matplotlib.pyplot as plt
from packing_graph import PackingGraph
from ant import Ant
import config as conf
import plotter
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

    
    # Get the weights of all bins
    bin_weights = list(best_ant.getBinWeights().values())
    # Calculate the average (mean) weight across all bins
    average_weight = sum(bin_weights) / len(bin_weights)
    # Calculate the difference of each bin weight from the average weight
    differences = [abs(weight - average_weight) for weight in bin_weights]
    # Compute the average difference
    average_difference = sum(differences) / len(differences)

    # Calculate load balance ratio
    max_weight = max(bin_weights)
    min_weight = min(bin_weights)
    # Calculate load balance ratio
    load_balance_ratio = max_weight / min_weight
    
    # Plot a bar chart of the bin weight distribution
    if conf.PLOT_WEIGHT_DIST: plotter.plotWeightDistribution(bin_weights)

    # Outputs
    print("Best fitness: ", best_fitness)
    print("Mean Absolute Deviation: ", abs(average_difference))
    print("Load balance ratio: ", f"{load_balance_ratio:.3g}")

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

if __name__ == "__main__":
    
    # Gets the bin num and items for the set problem
    bins, items = getValuesForProblem(conf.BPP1)   
    all_fitnesses = []

    # For each of the experiments
    for values in conf.P_AND_E_VALUES:
        print("Experiment: ", values)
        experiment_fitnesses = []
        fitness_progressions = []
        # Runs a set amount of trials
        for i in range(0, conf.NUM_TRIALS):
            print("Trial ", i+1)
            # i is used as the random seed for each trial
            fitness_prog, best_fitness = runExperiment(values[0], values[1], bins, items, i)

            fitness_progressions.append(fitness_prog)
            experiment_fitnesses.append(best_fitness)
            # Could do range / sum as well
        
        all_fitnesses.append(experiment_fitnesses)

        if conf.PLOT_FITNESS_PROGRESSION: plotter.plotBestFitnessProgression(fitness_progressions)
    
    if conf.PLOT_EXPERIMENT_TRIALS: plotter.plotExperimentTrials(all_fitnesses)
    

        