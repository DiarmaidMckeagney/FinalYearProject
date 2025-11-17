from torch_geometric.datasets import DBLP
from torch_geometric.transforms import AddMetaPaths


# right now this is just a test of the metapaths for the GNN model, nothing more. THis file will become a central file for running model(s)
data = DBLP(root="/home/diarmaid/Documents")[0]
# 4 node types: "paper", "author", "conference", and "term"
# 6 edge types: ("paper","author"), ("author", "paper"),
#               ("paper, "term"), ("paper", "conference"),
#               ("term, "paper"), ("conference", "paper")

# Add two metapaths:
# 1. From "paper" to "paper" through "conference"
# 2. From "author" to "conference" through "paper"
metapaths = [[("paper", "conference"), ("conference", "paper")],
             [("author", "paper"), ("paper", "conference")]]
data = AddMetaPaths(metapaths)(data)

print(data.edge_types)
print(data.metapath_dict)
