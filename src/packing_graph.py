""" Class for the construction graph, used to represent all possible decisions ants can make, when traversing
to each bin/item combination. Each directed edge also contains a pheremone value.
"""
import random

class PackingGraph():
    def __init__(self, num_bins, items, random_seed):
        """ Initialisation

        Args:
            num_bins int: The number of bins in the problem
            items int[]: The items to place in the bins
        """
        self.num_bins = num_bins
        self.items = items
        self.num_items = len(items)
        self.graph = {}
        self.random_seed = random_seed

    def initialiseGraph(self):
        """ Initalises the graph ready for the ACO algorithm

        Args:
            random_seed int: The seed for the random number generator
        """
        random.seed(self.random_seed)

        # Create a start and end node
        self.graph['Start'] = {'edges': {}}
        self.graph['End'] = {'edges': {}}

        # Create a node for each item in each possible bin
        for item in self.items:
            for bin in range(1, (self.num_bins + 1)):
                self.addNode(bin, item)

        # Connect Start to all nodes to the first item/bin combinations
        first_item = self.items[0]
        for bin in range(1, self.num_bins + 1):
            self.addEdge('Start', (bin, first_item), random.random())

        # Connect each node of current item to all nodes of the next item
        for i in range(self.num_items - 1):
            current_item = self.items[i]
            next_item = self.items[i + 1]

            # For each bin, connect current item to next item in any bin
            for current_bin in range(1, self.num_bins + 1):
                for next_bin in range(1, self.num_bins + 1):
                    self.addEdge((current_bin, current_item), (next_bin, next_item), random.random())

        # Connect all last bin/item combinations to End
        last_item = self.items[-1]
        for bin in range(1, self.num_bins + 1):
            self.addEdge((bin, last_item), 'End', 1) # Probabiltiy of going to End is 1, as it is the only possible node

    
    def addNode(self, bin_num, item):
        """Adds a new item, using the tuple (bin_num, item) as its ID

        Args:
            bin_num (int): The number of the bin
            item (int): The item being put in the bin
        """
        node_id = (bin_num, item)
        if node_id not in self.graph:
            self.graph[node_id] = {'edges': {}}

    def addEdge(self, from_id, to_id, pheromone):
        """ Adds a new path to the graph, with a pheromone weight

        Args:
            from_id (tuple): Node that the ant began at (bin, item)
            to_id (tuple): Node that the ant arrived at (bin, item)
            pheromone (int): Pheremone on the edge
        """
        if from_id in self.graph and to_id in self.graph:
            self.graph[from_id]['edges'][to_id] = pheromone
        else:
            print("One of the nodes is not in the graph")

    def updatePheromone(self, from_id, to_id, new_pheromone):
        """ Updates the pheromone on the directed edge

        Args:
            from_id (tuple): Node that the ant began at (bin, item)
            to_id (tuple): Node that the ant arrived at (bin, item)
            new_pheromone (int): Pheremone on the new edge
        """
        if from_id in self.graph and to_id in self.graph[from_id]['edges']:
            self.graph[from_id]['edges'][to_id] += new_pheromone # NEED TO CHECK IF ITS +=

    def evaporatePheromones(self, evaporation_rate):
        """ Evaporates pheromones on all edges in the graph.

        Args:
            evaporation_rate (float): The rate at which pheromones evaporate (0 < evaporation_rate < 1)
        """
        for from_node in self.graph:
            for to_node in self.graph[from_node]['edges']:
                self.graph[from_node]['edges'][to_node] *= (1-evaporation_rate) # More effective evapouration method

    def getEdges(self, node_id):
        """ Gets all the edges connected to the current node

        Args:
            node_id (tuple): Node that the ant is currently at (bin, item)
        Returns:
            dict : Nodes that are connected to a node, with their pheromone value
        """
        return self.graph[node_id]['edges']

    def getPheromone(self, from_id, to_id):
        """ Gets the pheremone for a specific edge between two nodes

        Args:
            from_id (tuple): Node the ant is currently at
            to_id (tuple): Node the ant may travel to
        Returns:
            float : Pheromone value for an edge
        """
        return self.graph[from_id]['edges'][to_id]

    def displayGraph(self):
        """ Outputs the packing graph
        """
        print("Graph:")
        for node, data in self.graph.items():
            if node == 'Start':
                print(f"Node {node}: Edges={data['edges']}")
            elif node == 'End':
                print(f"Node {node}")
            else:
                print(f"Node (Bin {node[0]}, Item {node[1]}): Edges={data['edges']}")

