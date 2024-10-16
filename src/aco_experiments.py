""" Script for running the different aco experiments on the 1DBPP
"""
from packing_graph import PackingGraph
from ant import Ant

def run_experiment(p, e, bins, items, random_seed):
    
    # STOP after 10,000 fitness evaluations
    no_of_evaluations = 0
    best_fitness = float('inf')
    best_path = []

    # Generate a graph
    graph = PackingGraph(bins, items)
    graph.initialiseGraph(random_seed)

    while no_of_evaluations < 10000:
        
        current_ants = []
        # 0-p ants traverse the graph
        for _ in range(0, p):
            # Traverse the graph
            ant = Ant(graph)
            ant.traverseGraph()
            # Store the Ant
            current_ants.append(ant)
            # print(ant.getPath())
            # print(ant.getFitness())
            # Get the fitness and see if its better than current best 
            if ant.getFitness() < best_fitness:
                best_fitness = ant.getFitness()
                best_path = ant.getPath()
                best_ant = ant
                # print(best_fitness)
            
            no_of_evaluations += 1

        # Update the pheromone paths by looping over list of ants and running the method
        for ant in current_ants:
            ant.updatePathPheromones()
        # Evaporate the pheromone
        graph.evaporatePheromones(e)
        # NEED TO MODIFY THE EVAPOURATION RATE!
    
    print(best_ant.getBinWeights())

    sorted_keys = sorted(best_ant.getBinWeights().keys())
    sorted_values = [best_ant.getBinWeights()[key] for key in sorted_keys]

    # Calculate differences between consecutive values
    differences = [sorted_values[i+1] - sorted_values[i] for i in range(len(sorted_values) - 1)]

    # Calculate the average of the differences
    average_difference = sum(differences) / len(differences)

    print("Average Difference:", abs(average_difference))

    # graph.displayGraph()
    # print(best_fitness)
    # print(best_path)


if __name__ == "__main__":
    # Create items and bins
    bins = 10
    items = []
    for i in range(1,500):
        items.append(i)

    run_experiment(100, 0.9, bins, items, 14)
    # Could do range / sum as well