""" ECM3401 Coursework README

This script contains all of the functionality for the ACO BPP coursework

How to run:


"""
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
""" Global variables for ACO experimentation
"""
# Experiment Variables
NUM_EVALUATIONS = 10000
NUM_TRIALS = 5
FITNESS_NUMERATOR = 100
P_AND_E_VALUES = [(10,0.6)]#[(100, 0.9) , (100, 0.6), (10, 0.9), (10, 0.6)]

# Problem names
BPP1 = 'BPP1'
BPP2 = 'BPP2'

PROBLEM_TO_SOLVE = BPP2

# Graph plotting
PLOT_BEST_FITNESS_PROGRESSION = True
PLOT_FITNESS_PROGRESSION_PER_P = True
PLOT_EXPERIMENT_TRIALS = True
PLOT_WEIGHT_DIST = False

""" Class for the construction graph, used to represent all possible decisions ants can make, when traversing
to each bin/item combination. Each directed edge also contains a pheremone value.
"""
class PackingGraph():
    def __init__(self, num_bins, items):
        """ Initialisation

        Args:
            num_bins int: The number of bins in the problem
            items int[]: The items to place in the bins
        """
        self.num_bins = num_bins
        self.items = items
        self.num_items = len(items)
        self.graph = {}

    def initialiseGraph(self):
        """ Initalises the graph ready for the ACO algorithm

        Args:
            random_seed int: The seed for the random number generator
        """
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
            self.addEdge((bin, last_item), 'End', 1) # Probabiltiy is 1, as it is the only possible node

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
        self.graph[from_id]['edges'][to_id] += new_pheromone

    def evaporatePheromones(self, evaporation_rate):
        """ Evaporates pheromones on all edges in the graph.

        Args:
            evaporation_rate (float): The rate at which pheromones evaporate (0 < evaporation_rate < 1)
        """
        for from_node in self.graph:
            for to_node in self.graph[from_node]['edges']:
                self.graph[from_node]['edges'][to_node] *= evaporation_rate

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

"""
Class declaration for an Ant, which will traverse across the generated graph based upon pheromones

"""
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
                current_item = next(iter(next_items.keys()))
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
            self.graph.updatePheromone(self.path[i], self.path[i+1], pheromone_value)
    
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

""" Data visualisation functions
"""
def plotBestFitnessProgressionAllTrials(best_fitnesses, title):
    """ Line graph of how the best fitness evolves over each of the trials for an experiment.

    Args:
        best_fitnesses (int[][]): List of the best fitnesses for each trial.
    """
    # Number of tests that have occurred
    test_numbers = range(len(best_fitnesses[0]))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Loop over each trial in best_fitnesses and plot them
    for idx, trial_best_values in enumerate(best_fitnesses):
        plt.plot(test_numbers, trial_best_values, linestyle='-', label=f'Trial {idx + 1}')

    # Add title and labels
    plt.title(title)
    plt.xlabel('Test Number')
    plt.ylabel('Fitness')

    # Format y-axis ticks with commas for thousands
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Show legend
    plt.legend()

    # Display the plot
    plt.grid()  # Optional: Add a grid for better readability
    plt.tight_layout()  # Adjust layout to fit title/labels
    plt.show()

def plotBestFitnessProgressionOneTrial(best_fitnesses):
    """ Line graph of how the best fitness evolves over a trial

    Args:
        best_fitnesses (int[]): List of the best fitnesses for a single trial
    """
    # Use the indices of the array as the test numbers
    test_numbers = range(len(best_fitnesses))  # Length of the best_fitnesses array

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the best fitness values for the single trial
    plt.plot(test_numbers, best_fitnesses, color=plt.cm.viridis(0.5), linestyle='-', label='Trial 1')

    # Add title and labels
    plt.title('Best Fitness Progression')
    plt.xlabel('Test Number')
    plt.ylabel('Fitness')

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()

def plotExperimentTrials(data):
    """
    Plots a grouped bar chart for multiple experiments, each with multiple trials.
    
    Parameters:
    - data (int[][]): List of lists, where each inner list contains fitness values for each trial in an experiment.
    """
    num_experiments = len(data)
    num_trials = len(data[0]) if num_experiments > 0 else 0
    
    experiment_labels = ["100, 0.9", "100, 0.6", "10, 0.9", "10, 0.6"]
    
    # Plotting parameters
    bar_width = 0.15
    space_between_experiments = 0.2
    
    # X positions for each group
    x_positions = np.arange(num_experiments) * (num_trials * bar_width + space_between_experiments)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting each trial within each experiment
    for i in range(num_trials):
        trial_positions = x_positions + i * bar_width
        trial_data = [experiment[i] for experiment in data]  # Extract each trial's data across experiments
        ax.bar(trial_positions, trial_data, width=bar_width, label=f'Trial {i+1}')
    
    # Customizing the plot
    ax.set_xticks(x_positions + (num_trials - 1) * bar_width / 2)
    ax.set_xticklabels(experiment_labels)
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Results for Each Experiment and Trial")
    # ax.set_yscale('log')
    ax.legend(title="Trials")
    plt.show()

def plotWeightDistribution(weights):
    """
    Plot a bar graph for the distribution of weights.

    Parameters:
    weights (int[]): An array of weights for each bin.
    """
    
    # Create an array of indices for the x-axis
    indices = list(range(len(weights)))

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(indices, weights, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Bin Number')
    plt.ylabel('Weight')
    plt.title('Weight Distribution of Bins')
    plt.xticks(indices)  # Set x-ticks to be the indices of weights
    plt.grid(axis='y')

    # Show the plot
    plt.show()

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
    graph = PackingGraph(bins, items) 
    graph.initialiseGraph()

    random.seed(random_seed)

    current_ants = [Ant(graph) for _ in range(num_ants)]

    # Holds best and worst fitness for each p paths
    fitness_prog = []
    best_fitness_per_p = []

    while no_of_evaluations < NUM_EVALUATIONS:
        best_fitness_for_p = float('inf')
        for ant in current_ants:
            # Ant traverses the graph
            current_fitness = ant.traverseGraph()
            # Check and store the best fitness overall
            if current_fitness < best_fitness_overall:
                best_fitness_overall = current_fitness
                best_ant = ant
            elif current_fitness < best_fitness_for_p:
                best_fitness_for_p = current_fitness
            
            no_of_evaluations += 1

        # Store the best fitness for the paths evaluated
        fitness_prog.append(best_fitness_overall)
        best_fitness_per_p.append(best_fitness_for_p)

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
    
    # Plot a bar chart of the bin weight distribution for a trial
    if PLOT_WEIGHT_DIST: plotWeightDistribution(bin_weights)

    # Outputs
    print("Best fitness: ", best_fitness_overall)
    print("Mean Absolute Deviation: ", abs(average_difference))
    print("Load balance ratio: ", f"{load_balance_ratio:.3g}")

    return fitness_prog, best_fitness_overall, best_fitness_per_p

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
    all_fitnesses = []

    # For each of the experiments
    for values in P_AND_E_VALUES:
        print("Experiment: ", values)
        experiment_fitnesses = []
        fitness_progressions = []
        fitness_progressions_per_p = []
        # Runs a set amount of trials
        for i in range(0, NUM_TRIALS):
            print("Trial ", i+1)
            # i is used as the random seed for each trial
            fitness_prog, best_fitness, best_fitness_per_p = runExperiment(values[0], values[1], bins, items, i)

            fitness_progressions.append(fitness_prog)
            experiment_fitnesses.append(best_fitness)
            fitness_progressions_per_p.append(best_fitness_per_p)
        
        all_fitnesses.append(experiment_fitnesses)

        # Plots the improvements to the best fitness
        if PLOT_BEST_FITNESS_PROGRESSION: plotBestFitnessProgressionAllTrials(fitness_progressions, "Best Fitness Progression")
        # Plots the recorded best fitness for each p ant paths
        if PLOT_FITNESS_PROGRESSION_PER_P: plotBestFitnessProgressionAllTrials(fitness_progressions_per_p, "Best Fitness For p Ant Paths")
    
    if PLOT_EXPERIMENT_TRIALS: plotExperimentTrials(all_fitnesses)
