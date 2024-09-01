import networkx as nx
import numpy as np

import parsing
from StochasticGradientDescent import sequential_stochastic_gradient_descent
from utils import find_best_threshold_rank_amazon

G = parsing.amazon_parse_edges("Amazon0601.txt", limit_nodes=1000)
# nx.draw(G, with_labels=True)
# plt.show()
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.size()}")

best_threshold, best_rank = find_best_threshold_rank_amazon(G, starting_rank=50, rank_step=50)
Z = sequential_stochastic_gradient_descent(G, best_rank, 0.001)
synthetic_adjacency_matrix = Z @ Z.T
synthetic_adjacency_matrix_with_threshold = np.where(synthetic_adjacency_matrix > best_threshold, 1, 0)

edges_that_dont_match = nx.to_numpy_array(G) != synthetic_adjacency_matrix_with_threshold
print(edges_that_dont_match)
num = 0
num_of_positives = 0
coords_of_recommendations = []
for i, row in enumerate(edges_that_dont_match):
    for j, col in enumerate(row):
        if col:
            num += 1
            if i != j and synthetic_adjacency_matrix_with_threshold[i][j] == 1:
                num_of_positives += 1
                coords_of_recommendations.append([i, j])

print(f"number of edges that dont match: {num}")
print(f"number of edges that dont match that are also one: {num_of_positives}")

with open("amazon_recommendations.txt", "w") as f:
    for i, j in coords_of_recommendations:
        f.write(f"{i}\t{j}\n")