import networkx as nx
import numpy as np

import StochasticGradientDescent
import parsing
from utils import find_best_threshold_rank

G = nx.DiGraph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (3, 4)])
print(nx.adjacency_matrix(G).todense())
Z = StochasticGradientDescent.sequential_stochastic_gradient_descent(G, 2)
Y = Z @ Z.T
print(Y)

best_threshold, _, best_precision = find_best_threshold_rank(G, 2)
Y_with_threshold = np.where(Y > best_threshold, 1, 0)
print(Y_with_threshold)
print(f"{best_precision * 100}%")