""" 
For generating the graph which will be used to apply ACO to the 1DBPP
"""

class PackingGraph():
    def __init__(self, numBins, items):
        """ Initialisation

        Args:
            numBins int: The number of bins in the problem
            items int[]: The items to place in the bins
        """
        self.numBins = numBins
        self.items = items
        self.numItems = len(items)
        self.graph = {}

    def initialiseGraph():

        # Create a start and end node

        # For each node, starting with the start node, create
        # numBins amount of nodes, which each contain value of the
        # current item index and add that to the list of nodes each node
        # has
        x = 1