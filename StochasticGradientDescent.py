import math

import networkx as nx
import numpy as np


def sequential_stochastic_gradient_descent(G: nx.Graph, r: int, eps: float = 0.001):
    """
    :param G: The graph to be factorized
    :param r: is the desired rank
    :param eps: is the error tolerance
    :return:
    """

    # Error checking
    if len(G.nodes) < r:
        raise ValueError("The rank cannot be greater than the number of nodes")
    if eps <= 0:
        raise ValueError("The error tolerance must be positive")
    if r <= 0:
        raise ValueError("The rank must be positive")

    # Initialize the matrix Z'
    # through testing, I found that initializing the matrix with small values creates a more consistent result
    # NOTE: not more accurate, but more consistent
    Z_prev = np.random.rand(len(G.nodes), r) * 0.1
    t = 1
    l = 1e-2
    Z = Z_prev.copy()
    while True:
        Z_prev = Z.copy()
        for node in G.nodes:
            gradient_total = np.zeros(r)
            for neighbor in G.neighbors(node):
                t += 1
                i = node
                j = neighbor
                # If we are dealing with a weighted graph, we need to consider the weight of the edge
                # but if the graph is unweighted, we can just assume the weight is 1
                Y_ij = G[i][j]['weight'] if 'weight' in G[i][j] else 1
                partial_gradient = (Y_ij - np.dot(Z[i], Z[j])) * Z[j]
                gradient_total += partial_gradient
            h = 1 / math.sqrt(t)
            Z[node] = Z[node] + h * (gradient_total - l * Z[node])
        #     print(f"Z[{node}, :] = {Z[node, :]}")
        # print(f'Norm: {np.linalg.norm(Z - Z_prev, ord="fro")}')
        if np.linalg.norm(Z - Z_prev, ord="fro") <= eps:
            break

    return Z