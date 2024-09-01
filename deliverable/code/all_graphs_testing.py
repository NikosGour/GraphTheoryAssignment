import os
import parsing
from utils import find_best_threshold_rank

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "data"

files = ["Amazon0302.txt", "Amazon0312.txt", "Amazon0505.txt", "Amazon0601.txt", "Email-EuAll.txt"]
precisions = []
thresholds = []
ranks = []
for file in files:
    G = parsing.amazon_parse_edges(file, limit_nodes=1000)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.size()}")

    best_threshold, best_rank, best_precision = find_best_threshold_rank(G, starting_rank=50, rank_step=50)
    precisions.append(best_precision)
    thresholds.append(best_threshold)
    ranks.append(best_rank)

for i, file_name in enumerate(files):
    print(
        f"The precision of file \"{file_name}\" is: {precisions[i] * 100}%. Best threshold: {thresholds[i]}. Best rank: {ranks[i]}")