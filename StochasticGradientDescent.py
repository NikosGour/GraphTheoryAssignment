import math

import networkx as nx
import numpy
import numpy as np




def sequential_stochastic_gradient_descent(Y: numpy.ndarray, r: int, eps: float):
    """
    :param Y: is the adjacency matrix of the graph, IN DENSE FORMAT
    :param r: is the desired rank
    :param eps: is the error tolerance
    :return:
    """

    # Error checking
    if Y.shape[0] < r:
        raise ValueError("The rank cannot be greater than the number of nodes")
    if eps <= 0:
        raise ValueError("The error tolerance must be positive")
    if r <= 0:
        raise ValueError("The rank must be positive")
    if Y.shape[0] != Y.shape[1]:
        raise ValueError("The adjacency matrix must be square")
    if not np.allclose(Y, Y.T):
        raise ValueError("The adjacency matrix must be symmetric")

    # Initialize the matrix Z'
    Z_prev = np.random.random((Y.shape[0], r))
    t = 1
    G = nx.from_numpy_array(Y)
    # G = G.to_directed()
    Z = Z_prev.copy()
    while True:
        Z_prev = Z.copy()
        for node in G.nodes():
            for neighbor in G.neighbors(node):
                h = 1 / math.sqrt(t)
                t += 1
                l = 1e-2
                i = node
                j = neighbor
                Z[i] = Z[i] + h * ((Y[i, j] - np.dot(Z[i], Z[j])) * Z[j] + l * Z[i])
                print(f"Z[{i}, :] = {Z[i, :]}")
        print(f'Norm: {np.linalg.norm(Z - Z_prev, ord="fro")}')
        if np.linalg.norm(Z - Z_prev, ord="fro") <= eps:
            break

    return Z

