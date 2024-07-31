import networkx as nx
import numpy as np

from StochasticGradientDescent import sequential_stochastic_gradient_descent


def find_best_threshold_rank(G: nx.Graph):
    best_precision = 0
    best_threshold = 0
    best_rank = 0

    for rank in range(2, int(G.number_of_nodes() / 2 + 1)):

        Z = sequential_stochastic_gradient_descent(G, rank)
        synthetic_adjacency_matrix = Z @ Z.T

        # We will iterate over a range of thresholds to find the best one
        for threshold in np.arange(0, 3, 0.01):
            # we need to apply a threshold to the synthetic adjacency matrix, because the original graph has 0s and 1s
            # and the synthetic matrix has floats. By applying a threshold, we can "interpret" the synthetic matrix as a
            # matrix of 0s and 1s
            synthetic_adjacency_matrix_with_thresholded = np.where(synthetic_adjacency_matrix > threshold, 1, 0)

            # Finding the precision of the factorized matrix
            nodes_that_match = nx.to_numpy_array(G) == synthetic_adjacency_matrix_with_thresholded
            precision = np.sum(nodes_that_match) / (G.number_of_nodes() * G.number_of_nodes())

            if precision > best_precision:
                best_precision = precision
                best_threshold = threshold
                best_rank = rank

    print(f"The best threshold is: {best_threshold}")
    print(f"The best rank is: {best_rank}")

    print(
        f"The precision of the factorized matrix is: {best_precision * 100}%")
    return best_threshold, best_rank