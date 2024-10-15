"""
An ant needs to - Traverse the graph, From the node it is currently at randomly pick a next node until at start
An ant need to - Store its current path

"""
from packing_graph import PackingGraph
import random

class Ant():
    def __init__(self, graph):
        """Initialisation

        Args:
            graph (PackingGraph): The graph the ant is going to traverse
        """
        self.graph = graph
        self.fitness = -1
        self.path = []

    def traverseGraph(self):
        """ Ant travels through the graph from Start to End, uses cummulative probability to select its next node

        """
        path = []
        # Starts at the Start node
        current_item = 'Start'
        bin_weights = {}

        # Loop until current node equals end
        while current_item != 'End':
            path.append(current_item)
            # Gets all the possible next routes
            next_items = self.graph.getEdges(current_item)
            # Checks if preceding the end node NEED TO CHECK THIS AT SOME POINT
            if len(next_items) > 1 :
                # Randomly selects an item, based upon pheromones
                total_pheromone = sum(next_items.values())
                random_number = random.uniform(0, total_pheromone)
                cumulative_pheromone = 0
                for item, pheromone in next_items.items():
                    cumulative_pheromone += pheromone
                    if random_number <= cumulative_pheromone:
                        current_item = item
                        break

                bin_num, weight = current_item
                # Update the bin weights as the ant travels
                if bin_num in bin_weights:
                    bin_weights[bin_num] += weight
                else:
                    bin_weights[bin_num] = weight
            else:
                current_item = 'End'
        
        # Calculate the fitness now the path has been complete
        fitness = self.calculateFitness(bin_weights)

        self.path = path
        self.fitness = fitness
    
    def calculateFitness(self, bin_weights):
        """ Calculates the difference between the most and least full bin

        Args:
            bin_weights (dict): Dictionary of all the bins and their current weights
        Returns:
            int : The fitness of the path
        """

        heaviest_bin = max(bin_weights.values())
        lightest_bin = min(bin_weights.values())
        return (heaviest_bin - lightest_bin)

    def updatePathPheromones(self):
        """ Updates the pheromone for the whole of the path the ant took
        """
        for i in range(0, len(self.path)-1):
            self.graph.updatePheromone(self.path[i], self.path[i+1], 100 / self.fitness)
    
    def getFitness(self):
        """ Returns the fitness of the ant

        Returns:
            int: Fitness value
        """
        return self.fitness
    
    def getPath(self):
        """ Returns the path of the ant

        Returns:
            tuple[]: Path of the ant
        """
        return self.path
