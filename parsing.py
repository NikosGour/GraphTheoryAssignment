import os
import networkx as nx
import glob

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "data"

def parse_edges(dir_name):
    file_path = os.path.join(project_dir, data_dir, dir_name)
    all_files = glob.glob(file_path + "/*.edges")
    G = nx.Graph()
    for i in range(4039):
        G.add_node(i)

    for file in all_files:
        with open(file, "r") as f:
            ego_node = int(file.split("\\")[-1].split(".")[0])
            for line in f:
                line = line.strip()
                line = line.split()
                G.add_edge(ego_node, int(line[0]))
                G.add_edge(ego_node, int(line[1]))
                G.add_edge(int(line[0]), int(line[1]))


    nx.write_gml(G, os.path.join(project_dir, data_dir, dir_name + ".gml"))





