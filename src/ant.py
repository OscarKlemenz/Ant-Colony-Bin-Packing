"""
An ant needs to - Traverse the graph, From the node it is currently at randomly pick a next node until at start
An ant need to - Store its current path

"""
from packing_graph import PackingGraph
import random

class Ant():
    def __init__(self, graph, random_seed):
        """Initialisation

        Args:
            graph (PackingGraph): The graph the ant is going to traverse
        """
        self.graph = graph
        random.seed(random_seed) # NOT SURE ABOUT THIS??

    def traverseGraph(self):
        """ Ant travels through the graph from Start to End, uses cummulative probability to select its next node

        """
        path = []
        # Starts at the Start node
        current_item = 'Start'
        # Loop until current node equals end
        while current_item != 'End':
            
            path.append(current_item)

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
            else:
                current_item = 'End'
        
        print(path)

graph = PackingGraph(3, [1, 2, 3])
graph.initialiseGraph(6)
graph.displayGraph()
ant = Ant(graph, 6)
ant.traverseGraph()