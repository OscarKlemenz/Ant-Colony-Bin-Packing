import random

""" Global variables for ACO experimentation
"""
# Experiment Variables
NUM_EVALUATIONS = 10000
NUM_TRIALS = 5
FITNESS_NUMERATOR = 100
P_AND_E_VALUES = [(100, 0.9)] #, (100, 0.6), (10, 0.9), (10, 0.6)]

# Problem names
BPP1 = 'BPP1'
BPP2 = 'BPP2'

PROBLEM_TO_SOLVE = BPP2

# Graph plotting
PLOT_BEST_FITNESS_PROGRESSION = True
PLOT_FITNESS_PROGRESSION_PER_P = True
PLOT_EXPERIMENT_TRIALS = True
PLOT_WEIGHT_DIST = False

class Graph():
    def __init__(self) -> None:
        pass
        self.matrix = None

    def initialiseGraph(self, n_bins, n_items):
        # Initialize the 3D weight matrix using nested lists
        pheromone_matrix = []

        # Weights from input layer to first hidden layer
        start = [[random.random() for _ in range(n_bins)] for _ in range(1)]
        pheromone_matrix.append(start)

        # Weights between hidden layers
        for _ in range(n_items - 1):
            item_layer = [[random.random() for _ in range(n_bins)] for _ in range(n_bins)]
            pheromone_matrix.append(item_layer)

        # Weights from the last hidden layer to output layer
        end = [[1 for _ in range(1)] for _ in range(n_bins)]
        pheromone_matrix.append(end)

        self.matrix = pheromone_matrix

    def updatePheromone(self, item, bin, next, new_pheromone):
        self.matrix[item][bin][next] += new_pheromone
    
    def getMatrix(self):
        return self.matrix
    
    def applyEvaporationRate(self, evaporation_rate):
        """ Evaporates pheromones on all edges in the graph.

        Args:
            evaporation_rate (float): The rate at which pheromones evaporate (0 < evaporation_rate < 1)
        """
        # Loop through each layer in the weight matrix
        for item in range(len(self.matrix)):
            # Loop through each row (node) in the layer
            for bin in range(len(self.matrix[item])):
                # Loop through each weight in the row
                for pheromone in range(len(self.matrix[item][bin])):
                    # Multiply the weight by the evaporation rate
                    self.matrix[item][bin][pheromone] *= evaporation_rate
    
    def getNextPheromones(self, item, bin):
        return self.matrix[item][bin]

    def getMatrixLen(self):
        return len(self.matrix)
    
    def displayMatrix(self):
        # Display the weight matrices
        print("Weights from Start to First Item:")
        for row in self.matrix[0]:
            print(row)

        for i in range(1, n_items):
            print(f"\nWeights from Item {i} to Item {i + 1}:")
            for row in self.matrix[i]:
                print(row)

        print("\nWeights from Last Item to End:")
        for row in self.matrix[n_items]:
            print(row)
    
class Ant():
    def __init__(self, graph, item_weights):
        """Initialisation

        Args:
            graph (PackingGraph): The graph the ant is going to traverse
        """
        self.graph = graph
        self.item_weights = item_weights
        self.fitness = -1
        self.path = []
        self.bin_weights = {}
    
    def traverseGraph(self):
        """ Ant travels through the graph from Start to End, selecting the next node using the random library
        """
        path = [(0,0)]
        bin_weights = {}

        # Start at the zeroth node
        current_bin = 0

        for item in range(0, self.graph.getMatrixLen()-1):
            
            # Checks if on the start node
            if item == 0:
                possible_paths = self.graph.getNextPheromones(item, 0)
            else:
                possible_paths = self.graph.getNextPheromones(item, current_bin)
            
            # Calculate cumulative probabilities
            total_pheromones = sum(possible_paths)
            cumulative_probabilities = []
            cumulative_sum = 0
            
            for pheromone in possible_paths:
                cumulative_sum += pheromone
                cumulative_probabilities.append(cumulative_sum / total_pheromones)  # Normalize
            
            # Generate a random number
            random_number = random.random()
            
            # Select the bin based on cumulative probabilities
            for bin_index, cumulative_prob in enumerate(cumulative_probabilities):
                if random_number < cumulative_prob:
                    current_bin = bin_index
                    break

            bin_weights[current_bin] = bin_weights.get(current_bin, 0) + self.item_weights[item]
            path.append((current_bin, item+1))

            
        # Calculate the fitness now the path has been complete
        fitness = self.calculateFitness(bin_weights)

        self.path = path
        self.fitness = fitness
        self.bin_weights = bin_weights

        return fitness
    
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
        pheromone_value = FITNESS_NUMERATOR / self.fitness
        for i in range(0, len(self.path)-1):
            self.graph.updatePheromone(self.path[i][1], self.path[i][0], self.path[i+1][0], pheromone_value)
    
    
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

""" Functions for running the different aco experiments on the 1DBPP
"""
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
    best_fitness_overall = float('inf')

    random.seed(99)
    # Generate a graph
    graph = Graph() 
    graph.initialiseGraph(bins, len(items))

    random.seed(random_seed)

    current_ants = [Ant(graph, items) for _ in range(num_ants)]

    while no_of_evaluations < NUM_EVALUATIONS:
        best_fitness_for_p = float('inf')
        for ant in current_ants:
            # Ant traverses the graph
            current_fitness = ant.traverseGraph()
            # Check and store the best fitness overall
            if current_fitness < best_fitness_overall:
                best_fitness_overall = current_fitness
                best_ant = ant
            
            no_of_evaluations += 1

        # Batch pheromone updates
        for ant in current_ants:
            ant.updatePathPheromones()
        
        # Evaporate the pheromones after all updates
        graph.applyEvaporationRate(evaporation_rate)

    # Outputs
    print("Best fitness: ", best_fitness_overall)
    return best_fitness_overall


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
    bins, items = getValuesForProblem(PROBLEM_TO_SOLVE)
    
    # For each of the experiments
    for values in P_AND_E_VALUES:
        print("Experiment: ", values)
        # Runs a set amount of trials
        for i in range(0, NUM_TRIALS):
            print("Trial ", i+1)
            # i is used as the random seed for each trial
            runExperiment(values[0], values[1], bins, items, i)




# # Set random seed for reproducibility
# random.seed(94)
# n_bins = 10
# n_items = 10
# n_ants = 5  # Number of ants

# graph = Graph()
# graph.initialiseGraph(n_bins, n_items)

# # Display the weight matrices
# graph.displayMatrix()

# items = list(range(1, 11))
# print(items)

# new_ant = Ant(graph, items)
# print(new_ant.traverseGraph())

# new_ant.updatePathPheromones()