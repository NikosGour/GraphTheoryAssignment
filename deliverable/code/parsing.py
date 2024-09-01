import os
import networkx as nx

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "data"


def amazon_parse_edges(file_name, limit_nodes=None) -> nx.DiGraph:
    file_path = os.path.join(project_dir, data_dir, file_name)
    G = nx.DiGraph()

    with open(file_path, "r") as f:
        lines = f.readlines()
        # First four lines are comment
        lines = lines[4:]
        for i, line in enumerate(lines):
            from_node, to_node = line.split("\t")
            from_node = int(from_node)
            to_node = int(to_node)
            if limit_nodes is not None:
                if G.number_of_nodes() < limit_nodes:
                    if from_node > limit_nodes or to_node > limit_nodes:
                        continue
                else:
                    break
            G.add_edge(from_node, to_node)
            # print(f"Saved edge {i}")

    # nx.write_gml(G, os.path.join(project_dir, data_dir, file_name + ".gml"))
    return G

# G = parse_edges("Amazon0601.txt")
# # nx.draw(G, with_labels=True)
# # plt.show()
# print("Done")