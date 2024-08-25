import os

import numpy as np
from matplotlib import pyplot as plt

import parsing
from StochasticGradientDescent import sequential_stochastic_gradient_descent
import networkx as nx

from utils import find_best_threshold_rank

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "data"

# G = nx.read_gml(os.path.join(project_dir, data_dir, "Amazon0601.txt.gml"))
G = parsing.amazon_parse_edges("Amazon0601.txt", limit_nodes=1000)
nx.draw(G, with_labels=True)
plt.show()
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.size()}")
# print(nx.adjacency_matrix(G).todense())
# Z = sequential_stochastic_gradient_descent(G, 2, 0.001)
# print(Z)
#
# synthetic_adjacency_matrix = Z @ Z.T

best_threshold, best_rank = find_best_threshold_rank(G, starting_rank=50, rank_step=50)
# best_rank = 150
# best_threshold = 1.06
Z = sequential_stochastic_gradient_descent(G, best_rank, 0.001)
synthetic_adjacency_matrix = Z @ Z.T
synthetic_adjacency_matrix_with_threshold = np.where(synthetic_adjacency_matrix > best_threshold, 1, 0)
nodes_that_dont_match = nx.to_numpy_array(G) != synthetic_adjacency_matrix_with_threshold
print(nodes_that_dont_match)
num = 0
num_of_positives = 0
coords_of_recommendations = []
for i, row in enumerate(nodes_that_dont_match):
    for j, col in enumerate(row):
        if col:
            num += 1
            if i != j and synthetic_adjacency_matrix_with_threshold[i][j] == 1:
                num_of_positives += 1
                coords_of_recommendations.append([i, j])

print(f"number of edges that dont match: {num}")
print(f"number of edges that dont match that are also one: {num_of_positives}")

# for i, j in coords_of_recommendations:
#     print(f"Item node{i} should be co-purchased with item node{j}")
# nx.draw(nx.from_numpy_array(synthetic_adjacency_matrix_with_thresholded), with_labels=True)
# plt.show()