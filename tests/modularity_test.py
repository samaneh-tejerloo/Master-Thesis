# %%
from train.utils import get_algorithm_complexes
from dataset import PPIDataLoadingUtil
import torch
from torch_geometric.data import Data
from models import SimpleGNN
import torch_geometric.nn as gnn

# %%

dataset = PPIDataLoadingUtil(
    "datasets/tadw-sc/krogan-core/krogan-core.csv",
    load_weights=True,
    load_embeddings=False,
)


features = dataset.get_features("one_hot", name_spaces=["MF", "BP"])


features = torch.tensor(features, dtype=torch.float32)
features.shape


edge_index = torch.LongTensor(dataset.edges_index).T
edge_weights = torch.tensor(dataset.weights)

data = Data(x=features, edge_index=edge_index)


model = SimpleGNN(data.num_features, 512, 512, 2, gnn.GATConv, heads=4)

weights = torch.load(
    "logs/weights/krogan-core_SimpleGNN_GAT_2-layers_4-heads_relu_BP_MF_512_2000.pt"
)
model.load_state_dict(weights)
model.eval()

with torch.no_grad():
    F_out = model(data)
# %%
clusters = get_algorithm_complexes(F_out, dataset, threshold=0.3)
# %%
clusters_id = []
for cluster in clusters:
    cluster_id = []
    for p in cluster:
        cluster_id.append(dataset.protein_name_to_id(p))
    clusters_id.append(cluster_id)
# %%
l_intra = 0
for cluster_id in clusters_id:
    c = len(cluster_id)
    denominator = c * (c - 1)
    sum = 0
    for i in cluster_id:
        for j in cluster_id:
            if i != j:
                sum += torch.norm(F_out[i] - F_out[j])
    l_intra += sum / denominator

l_intra /= len(clusters_id)
# %%
l_inter = 0
for cluster_id in clusters_id:
    c = len(cluster_id)
    n = len(dataset.proteins)
    denominator = c * (n - c)
    sum = 0
    for i in cluster_id:
        for j in range(n):
            if j not in cluser_id:
                sum += torch.norm(F_out[i] - F_out[j])
    l_inter += sum / denominator
l_inter /= len(clusters_id)
print(l_inter)
