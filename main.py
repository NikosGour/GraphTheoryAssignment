import os
from StochasticGradientDescent import sequential_stochastic_gradient_descent
import networkx as nx

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "data"

# G = nx.read_gml(os.path.join(project_dir, data_dir, "facebook.gml"))
G = nx.Graph()
G.add_edge(0, 1)
G.add_edge(0, 2)
G.add_edge(1, 3)
G.add_edge(2, 3)
G.add_edge(3, 4)
G.add_edge(4, 5)
G.add_edge(4, 6)
G.add_edge(5, 6)
G.add_edge(6, 7)
G.add_edge(6, 8)
G.add_edge(7, 8)
G.add_edge(8, 9)
G.add_edge(8, 10)
G.add_edge(9, 10)
G.add_edge(10, 11)
G.add_edge(10, 12)
res = sequential_stochastic_gradient_descent(nx.adjacency_matrix(G).todense(),2, 0.001)
print(res)
# possible min max scaling