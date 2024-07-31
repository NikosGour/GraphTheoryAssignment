import os

import numpy as np

from StochasticGradientDescent import sequential_stochastic_gradient_descent
import networkx as nx

from utils import find_best_threshold_rank

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "data"

# G = nx.read_gml(os.path.join(project_dir, data_dir, "facebook.gml"))
G = nx.Graph([(0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (3, 4)])
print(nx.adjacency_matrix(G).todense())
Z = sequential_stochastic_gradient_descent(G, 2, 0.001)
print(Z)

synthetic_adjacency_matrix = Z @ Z.T
# plot the graphs
import matplotlib.pyplot as plt

nx.draw(G, with_labels=True)
plt.show()

best_threshold, best_rank = find_best_threshold_rank(G)
synthetic_adjacency_matrix = Z @ Z.T
synthetic_adjacency_matrix_with_thresholded = np.where(synthetic_adjacency_matrix > best_threshold, 1, 0)

nx.draw(nx.from_numpy_array(synthetic_adjacency_matrix_with_thresholded), with_labels=True)
plt.show()