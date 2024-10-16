""" Script for running the different aco experiments on the 1DBPP
"""
from packing_graph import PackingGraph
from ant import Ant
import config as conf

def runExperiment(p, e, bins, items, random_seed):
    """ Runs a set amount of evaluations of the ACO algorithm on the 1DBPP

    Args:
        p (int): The number of paths to generate before updating the pheromone values
        e (float): Evaporation rate of the pheromones
        bins (int): The number of bins the items can be placed in
        items (int[]): The items to place in the bins
        random_seed (int): The seed value for when random is used in the experiment
    """
    
    # STOP after 10,000 fitness evaluations
    no_of_evaluations = 0
    best_fitness = float('inf')

    # Generate a graph
    graph = PackingGraph(bins, items)
    graph.initialiseGraph(random_seed)

    while no_of_evaluations < conf.NUM_EVALUATIONS:
        
        current_ants = []
        # 0-p ants traverse the graph
        for _ in range(0, p):
            # Traverse the graph
            ant = Ant(graph)
            ant.traverseGraph()
            # Store the Ant
            current_ants.append(ant)
            # Get the fitness and see if its better than current best 
            if ant.getFitness() < best_fitness:
                best_fitness = ant.getFitness()
                best_ant = ant
            
            no_of_evaluations += 1

        # Update the pheromone paths by looping over list of ants and running the method
        for ant in current_ants:
            ant.updatePathPheromones()
        # Evaporate the pheromone
        graph.evaporatePheromones(e)
        # NEED TO MODIFY THE EVAPOURATION RATE!
    
    # AVERAGE CALCULATIONS
    sorted_keys = sorted(best_ant.getBinWeights().keys())
    sorted_values = [best_ant.getBinWeights()[key] for key in sorted_keys]
    # Calculate differences between consecutive values
    differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values) - 1)]
    # Calculate the average of the differences
    average_difference = sum(differences) / len(differences)

    print("Best fitness: ", best_fitness)
    print("Average difference of bins for best fitness: ", abs(average_difference))

def getValuesForProblem(bin_problem):
    """ Returns the correct bin and item values for a set problem

    Args:
        bin_problem (str): The type of bin problem to undertake

    Returns:
        int, int[]: The number of bins and the items to be placed in the bins
    """

    items = []
    if bin_problem == "BPP1":
        for i in range(1,500):
            items.append(i)
        return 10, items
    elif bin_problem == "BPP2":
        for i in range(1,500):
            items.append(i**2/2)
        return 50, items
    else:
        return 0, items

if __name__ == "__main__":
    
    # Gets the bin num and items for the set problem
    bins, items = getValuesForProblem(conf.BPP1)

    # Runs a set amount of trials
    for i in range(0, conf.NUM_TRIALS):
        # i is used as the random seed for each trial
        runExperiment(100, 0.9, bins, items, i)
        # Could do range / sum as well