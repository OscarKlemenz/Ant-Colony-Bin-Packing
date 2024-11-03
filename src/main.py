""" ECM341 Coursework

This script implements an Ant Colony Optimization (ACO) algorithm for solving the Bin Packing Problem (BPP) 
as part of the ECM3412 coursework. 

The script includes configuration options for experimentation with various parameters, 
and generates visualisations of the algorithm across trials.

-- Prerequisites -- 

This script requires Python and the following libraries for plotting and numerical computations:
- matplotlib
- numpy

Install the dependencies using: pip install matplotlib numpy

-- Running the Script --

To execute the script, run the following command in the terminal: python3 main.py

-- Configuration for Experimentation --

Several global variables are provided to control the script behavior and enable experimentation:

- PROBLEM_TO_SOLVE - Defines the Bin Packing Problem (BPP) instance to be solved.

Experiment Parameters - Key variables that affect the behavior of the ACO algorithm:

- NUM_EVALUATIONS: Total number of evaluations to run for the algorithm (default: 10,000)
- NUM_TRIALS: Number of trials to perform for each experiment configuration (default: 5)
- FITNESS_NUMERATOR: Constant used in calculating pheromone levels based on fitness
- P_AND_E_VALUES: List of tuples (p, e) where p is the number of paths to generate before 
                  updating pheromones, and e is the evaporation rate

Graph Plotting - Flags to enable or disable various plots for visualising the results:

- PLOT_BEST_FITNESS_PROGRESSION: Plot best fitness progression across trials
- PLOT_FITNESS_PROGRESSION_PER_P: Plot best fitness progression per p value
- PLOT_EXPERIMENT_TRIALS: Plot bar chart of best fitness for each trial
- PLOT_WEIGHT_DIST: Plot bin weight distribution for a single trial

Further Experimentation - Experimentation that goes beyond the spec

- LOCAL_HEURSTIC_EXPERIMENT: Flag for using a local heuristic when deciding on the next
                             node (Improves BPP2 significantly)
- ALPHA: How much weight the heuristic has (0-1)
- VARIABLE_E_VALUES: Some different e values, to see how they impact the ACO
- VARIABLE_P_VALUES: Some different p values, to see how they impact the ACO

"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import random
from typing import List, Tuple
import sys
""" 
Global variables for ACO experimentation
"""
# Experiment Variables
NUM_EVALUATIONS = 10000
NUM_TRIALS = 5
FITNESS_NUMERATOR = 100
P_AND_E_VALUES = [(100, 0.9), (100, 0.6), (10, 0.9), (10, 0.6)]

# Problems
BPP1 = 'BPP1'
BPP2 = 'BPP2'

PROBLEM_TO_SOLVE = BPP2

# Graph plotting
PLOT_BEST_FITNESS_PROGRESSION = False
PLOT_FITNESS_PROGRESSION_PER_P = True
PLOT_EXPERIMENT_TRIALS = True
PLOT_WEIGHT_DIST = False

# Further Experimentation
LOCAL_HEURSTIC_EXPERIMENT = False
ALPHA = 0.1
VARIABLE_E_VALUES = [(10, 0.85), (10, 0.8), (10,0.75), (10, 0.7)]
VARIABLE_P_VALUES = [(5, 0.9), (20, 0.9), (50, 0.9), (75, 0.9)]
""" 
Class for the construction graph, used to represent all possible decisions ants can make, when traversing
to each bin/item combination. Each directed edge also contains a pheremone value.
"""
class Graph():
    def __init__(self, num_bins: int, items: List[int]) -> None:
        """ Initialisation

        Args:
            num_bins (int): The number of bins in the problem
            items (int[]): The items to place in the bins
        """
        self.num_bins = num_bins
        self.items = items
        self.num_items = len(items)
        self.graph = {}

    def initialiseGraph(self) -> None:
        """ Creates the graph ready for the ACO algorithm

        Args:
            random_seed (int): The seed for the random number generator
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

    def addNode(self, bin_num: int, item: int) -> None:
        """ Adds a new item, using the tuple (bin_num, item) as its ID

        Args:
            bin_num (int): The number of the bin
            item (int): The item being put in the bin
        """
        node_id = (bin_num, item)
        if node_id not in self.graph:
            self.graph[node_id] = {'edges': {}}

    def addEdge(self, from_id: Tuple[int, int], to_id: Tuple[int, int], pheromone: int) -> None:
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

    def updatePheromone(self, from_id: Tuple[int, int], to_id: Tuple[int, int], new_pheromone: int) -> None:
        """ Updates the pheromone on the directed edge

        Args:
            from_id (tuple): Node that the ant began at (bin, item)
            to_id (tuple): Node that the ant arrived at (bin, item)
            new_pheromone (int): Pheremone on the new edge
        """
        self.graph[from_id]['edges'][to_id] += new_pheromone

    def evaporatePheromones(self, evaporation_rate: float) -> None:
        """ Evaporates pheromones on all edges in the graph.

        Args:
            evaporation_rate (float): The rate at which pheromones evaporate (0 < evaporation_rate < 1)
        """
        for from_node in self.graph:
            for to_node in self.graph[from_node]['edges']:
                self.graph[from_node]['edges'][to_node] *= evaporation_rate

    def getEdges(self, node_id: Tuple[int, int]) -> dict:
        """ Gets all the edges connected to the current node

        Args:
            node_id (tuple): Node that the ant is currently at (bin, item)
        Returns:
            Nodes that are connected to a node, with their pheromone value
        """
        return self.graph[node_id]['edges']

    def displayGraph(self):
        """ Outputs the graph
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
    def __init__(self, graph: Graph, num_items: int) -> None:
        """ Initialisation

        Args:
            graph (Graph): The graph the ant is going to traverse
            num_items (int): Number of items, indicates how long ant path will be, used for phereomone
                             update
        """
        self.graph = graph
        self.fitness = -1
        self.path = []
        self.bin_weights = {}
        self.path_len = num_items
        # Initialise bin weights and target weight for heuristic (Further Work)
        self.target_weight = sum(self.graph.items) / self.graph.num_bins

    def traverseGraph(self) -> int:
        """ Ant travels through the graph from Start to End, selecting the next node using the random library

        Returns:
            int: The fitness of the traversal
        """
        path = []
        bin_weights = {}

        # Starts at the Start node
        current_item = 'Start'
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
                # For Further Experimentation, uses a heuristic based on how close bins are to optimal weight
                if LOCAL_HEURSTIC_EXPERIMENT:
                    adjusted_weights = []
                    for item, pheromone in next_items.items():
                        bin_num, weight = item
                        current_bin_weight = bin_weights.get(bin_num, 0)
                        
                        # The closer to target_weight, the weaker the pheromone should be
                        heuristic_value = 1 / (1 + abs(current_bin_weight + weight - self.target_weight))
                        
                        # Adjusted weight combines pheromone and heuristic
                        adjusted_weight = pheromone * (heuristic_value ** ALPHA)
                        adjusted_weights.append(adjusted_weight)
                        pheromones = adjusted_weights

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
    
    def calculateFitness(self, bin_weights: dict) -> int:
        """ Calculates the difference between the most and least full bin

        Args:
            bin_weights (dict): Dictionary of all the bins and their current weights
        Returns:
            int : The fitness of the path
        """
        heaviest_bin = max(bin_weights.values())
        lightest_bin = min(bin_weights.values())
        return (heaviest_bin - lightest_bin)

    def updatePathPheromones(self) -> None:
        """ Updates the pheromone for the whole of the path the ant took
        """
        pheromone_value = FITNESS_NUMERATOR / self.fitness
        for i in range(0, self.path_len):
            self.graph.updatePheromone(self.path[i], self.path[i+1], pheromone_value)
    
    def getBinWeights(self) -> dict:
        """ Returns the bin weights of the path the ant took

        Returns:
            dict: Bin weights of the path
        """
        return self.bin_weights

""" 
Data visualisation functions
"""
def plotBestFitnessProgressionAllTrials(best_fitnesses: List[List[int]], title: str) -> None:
    """ Line graph of how the best fitness evolves over each of the trials for an experiment.

    Args:
        best_fitnesses (int[][]): List of the best fitnesses for each trial.
        title (str): The title of the line graph
    """
    # Number of tests that have occurred
    test_numbers = range(len(best_fitnesses[0]))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Loop over each trial in best_fitnesses and plot them
    for idx, trial_best_values in enumerate(best_fitnesses):
        plt.plot(test_numbers, trial_best_values, linestyle='-', label=f'Trial {idx + 1}')

    # Title and labels
    plt.title(title)
    plt.xlabel('Evaluation Count (per p Paths)')
    plt.ylabel('Best Fitness')
    plt.legend()

    # Format the y-axis with commas for thousands using a lambda function
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.grid(linestyle='-')
    plt.tight_layout()
    plt.show()

def plotBestFitnessProgressionOneTrial(best_fitnesses: List[int]) -> None:
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

    # Title and labels
    plt.title('Best Fitness Progression')
    plt.xlabel('Test Number')
    plt.ylabel('Fitness')
    plt.legend()

    plt.show()

def plotExperimentTrials(data: List[List[int]]) -> None:
    """ Plots a grouped bar chart for multiple experiments, each with multiple trials.
    
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
    
    _, ax = plt.subplots(figsize=(10, 6))

    # Plotting each trial within each experiment
    for i in range(num_trials):
        trial_positions = x_positions + i * bar_width
        trial_data = [experiment[i] for experiment in data]  # Extract each trial's data across experiments
        ax.bar(trial_positions, trial_data, width=bar_width, label=f'Trial {i+1}', zorder=3)
    
    # Customisations
    ax.set_xticks(x_positions + (num_trials - 1) * bar_width / 2)
    ax.set_xticklabels(experiment_labels)
    ax.set_ylabel("Best Fitness")
    ax.set_title("Best Fitness Results for Each Experiment and Trial")
    ax.legend(title="Trials")

    # Format the y-axis with commas for thousands using a lambda function
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax.grid(which='both', linestyle='-', linewidth=0.7, zorder=1)
    plt.show()

def plotWeightDistribution(weights: List[int]) -> None:
    """ Plot a bar graph for the distribution of weights.

    Parameters:
    weights (int[]): An array of weights for each bin.
    """
    
    # Array of the indices of weights for the x-axis
    indices = list(range(len(weights)))

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(indices, weights, color='skyblue', edgecolor='black')

    # Labels and the title
    plt.xlabel('Bin Number')
    plt.ylabel('Weight')
    plt.title('Weight Distribution of Bins')
    plt.xticks(indices)  # Set x-ticks to be the indices of weights
    plt.grid(axis='y')

    plt.show()

""" 
Driver functions for running the aco experiments on the 1DBPP
"""
def runExperiment(num_ants: int, evaporation_rate: float, bins: int, items: List[int], random_seed: int) -> Tuple[int, int, int]:
    """ Runs a set amount of evaluations of the ACO algorithm on the 1DBPP

    Args:
        num_ants (int): The number of paths to generate before updating the pheromone values (p)
        evaporation_rate (float): Evaporation rate of the pheromones (e)
        bins (int): The number of bins the items can be placed in
        items (int[]): The items to place in the bins
        random_seed (int): The seed value for when random is used in the experiment
    Returns:
        The best fitness, the progression of the best fitness and the best fitness recorded per p paths
    """
    # Sets the seed
    random.seed(random_seed)

    no_of_evaluations = 0
    best_fitness_overall = float('inf')

    # Generate a graph
    graph = Graph(bins, items) 
    graph.initialiseGraph()

    # Ants to be used for traversal
    current_ants = [Ant(graph, len(items)) for _ in range(num_ants)]

    # Holds the best fitnesses overall
    fitness_prog = []
    # Holds best and worst fitnesses for each p paths
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
            # Store the best fitness for the current p paths
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

        # Output progress in place
        sys.stdout.write(f"\rTotal evaluations so far: {no_of_evaluations}/{NUM_EVALUATIONS}")
        sys.stdout.flush() 

    
    # Get the weights of all bins
    bin_weights = list(best_ant.getBinWeights().values())
    # Calculate the mean weight
    average_weight = sum(bin_weights) / bins
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
    print("\nBest fitness: ", best_fitness_overall)
    print("Mean Absolute Deviation: ", abs(average_difference))
    print("Load balance ratio: ", f"{load_balance_ratio:.3g}")

    return best_fitness_overall, fitness_prog, best_fitness_per_p

def getValuesForProblem(bin_problem: str) -> Tuple[int, List[int]]:
    """ Returns the correct bin and item values for a set problem

    Args:
        bin_problem (str): The type of bin problem to undertake
    Returns:
        (int, int[]): The number of bins and the items to be placed in the bins
    """
    items = []
    if bin_problem == "BPP1":
        items = list(range(1, 501))
        return 10, items
    elif bin_problem == "BPP2":
        items = [i ** 2 / 2 for i in range(1, 501)]
        return 50, items
    else:
        print("No values available for the problem")
        return 0, items

if __name__ == "__main__":
    
    # Gets the bin num and items for the set problem
    bins, items = getValuesForProblem(PROBLEM_TO_SOLVE)   
    # For plotting
    all_fitnesses = []

    # For each of the experiments
    for values in P_AND_E_VALUES:
        print("Experiment: ", values)
        # For plotting
        experiment_fitnesses, fitness_progressions, fitness_progressions_per_p = [], [], []
        # Runs a set amount of trials
        for i in range(0, NUM_TRIALS):
            print("Trial ", i+1)
            # i is used as the random seed for each trial
            best_fitness, fitness_prog, best_fitness_per_p = runExperiment(values[0], values[1], bins, items, i)

            fitness_progressions.append(fitness_prog)
            experiment_fitnesses.append(best_fitness)
            fitness_progressions_per_p.append(best_fitness_per_p)
        
        all_fitnesses.append(experiment_fitnesses)

        # Plots the improvements to the best fitness
        if PLOT_BEST_FITNESS_PROGRESSION: plotBestFitnessProgressionAllTrials(fitness_progressions, "Best Fitness Progression")
        # Plots the recorded best fitness for each p ant paths
        if PLOT_FITNESS_PROGRESSION_PER_P: plotBestFitnessProgressionAllTrials(fitness_progressions_per_p, "Best Fitness Recorded Across Evaluations")
    # Plots all the best fitnesses on a bar chart
    if PLOT_EXPERIMENT_TRIALS: plotExperimentTrials(all_fitnesses)
