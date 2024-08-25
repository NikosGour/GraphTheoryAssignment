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
files = ["Amazon0302.txt", "Amazon0312.txt", "Amazon0505.txt", "Amazon0601.txt", "Email-EuAll.txt"]
precisions = []
thresholds = []
ranks = []
for file in files:
    G = parsing.amazon_parse_edges(file, limit_nodes=1000)
    # nx.draw(G, with_labels=True)
    # plt.show()
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.size()}")
    # print(nx.adjacency_matrix(G).todense())
    # Z = sequential_stochastic_gradient_descent(G, 2, 0.001)
    # print(Z)
    #
    # synthetic_adjacency_matrix = Z @ Z.T

    best_threshold, best_rank, best_precision = find_best_threshold_rank(G, starting_rank=50, rank_step=50)
    precisions.append(best_precision)
    thresholds.append(best_threshold)
    ranks.append(best_rank)

for i, file_name in enumerate(files):
    print(
        f"The precision of file \"{file_name}\" is: {precisions[i] * 100}%. Best threshold: {thresholds[i]}. Best rank: {ranks[i]}")