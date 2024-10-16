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
        self.bin_weights = {}

    def traverseGraph(self):
        """ Ant travels through the graph from Start to End, selecting the next node using the random library
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
            
            if len(next_items) == 1:
                current_item = list(next_items.keys())[0]
            else:
                # Choose based on weighted pheromones
                items, pheromones = zip(*next_items.items())
                current_item = random.choices(items, weights=pheromones, k=1)[0]
                # Update bin weights based on the current item
                bin_num, weight = current_item
                bin_weights[bin_num] = bin_weights.get(bin_num, 0) + weight
        
        # Calculate the fitness now the path has been complete
        fitness = self.calculateFitness(bin_weights)

        self.path = path
        self.fitness = fitness
        self.bin_weights = bin_weights
    
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
    
    def getBinWeights(self):
        """ Returns the bin weights of the path the ant took

        Returns:
            dict: Bin weights of the path
        """
        return self.bin_weights
