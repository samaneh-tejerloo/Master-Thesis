# %%
from tqdm import tqdm
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from train.utils import get_algorithm_complexes
from dataset import PPIDataLoadingUtil
import torch
from torch_geometric.data import Data
from models import SimpleGNN
import torch_geometric.nn as gnn
import networkx as nx
from torch_geometric.utils import to_networkx

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
# %%
# checking networkx modularity implementation which does not work for overlapping clusters
nx_graph = to_networkx(data)
communities = [set(comp) for comp in clusters_id]
nx.community.modularity(nx_graph, communities)

G = nx.barbell_graph(3, 0)
nx.community.modularity(G, [{5, 1, 2}, {3, 4, 0, 5}])
# %%
# overlapping modularity extension but needs optimization
m = data.edge_index.shape[1] // 2
n = data.x.shape[0]

A = np.zeros((n, n))
A[data.edge_index[0], data.edge_index[1]] = 1

mod_sum = 0
for i in tqdm(range(n)):
    o_i = sum(list(map(lambda x: i in x, clusters_id)))
    k_i = A[i, :].sum()
    for j in range(n):
        delta = sum(list(map(lambda x: i in x and j in x, clusters_id)))
        if delta > 0:
            delta = 1
        else:
            continue
        o_j = sum(list(map(lambda x: j in x, clusters_id)))
        a_ij = A[i, j]
        k_j = A[j, :].sum()
        mod_sum += 1 / (o_i * o_j) * (a_ij - ((k_i * k_j) / (m * 2))) * delta
        # print(mod_sum)
modularity = mod_sum / (2 * m)
print(modularity)
# %%
homo_sum = 0
for cluster in tqdm(clusters_id):
    x = F_out[cluster, :]
    mu_c = x.mean(0, keepdim=True)
    homogeneity = np.linalg.norm(x - mu_c, axis=1).mean()
    homo_sum += homogeneity
homo = homo_sum / len(clusters_id)
print(homo)
# %%
# Silhouette score
all_si = []
for i in tqdm(range(n)):
    clusters_with_i = [cluster for cluster in clusters_id if i in cluster]
    # print(i, clusters_with_i)
    if len(clusters_with_i) == 0:
        continue

    clusters_without_i = [cluster for cluster in clusters_id if i not in cluster]

    a_is = []
    for cluster_with_i in clusters_with_i:
        # print(cluster_with_i)
        x = F_out[cluster_with_i, :]
        a_i = np.linalg.norm(x - F_out[i, :], axis=1).mean()
        a_is.append(a_i)
    a_i = np.mean(a_is)
    distances_inter = []
    for cluster_without_i in clusters_without_i:
        x = F_out[cluster_without_i, :]
        distance_inter = np.linalg.norm(x - F_out[i, :], axis=1).mean()
        distances_inter.append(distance_inter)
    b_i = min(distances_inter)
    s_i = (b_i - a_i) / max(a_i, b_i)

    all_si.append(s_i)
    # print(s_i)
print(np.mean(all_si))
# %%
# optimizated version of modularity calculation
threshold = 0.3
clustering = F_out > threshold
clustering = clustering.int().numpy()
n = data.x.shape[0]
m = data.edge_index.shape[1] // 2
A = np.zeros((n, n))
A[data.edge_index[0], data.edge_index[1]] = 1
degree_vector = A.sum(axis=0).reshape(-1, 1)
K = (degree_vector @ degree_vector.T) / (2 * m)

o_vector = clustering.sum(axis=1, keepdims=True)
O = o_vector @ o_vector.T

delta = (clustering @ clustering.T) > 0
O_safe = O.copy()
O_safe[O_safe == 0] = 1
modularity = 1 / (2 * m) * ((1 / O_safe) * (A - K) * delta).sum()
modularity
# %%
# Density
M = clustering
cluster_sizes = M.sum(axis=0)

edges_per_cluster = np.diag(M.T @ A @ M) / 2
denominator = cluster_sizes * (cluster_sizes - 1) / 2

density_vector = edges_per_cluster / denominator
density_vector[denominator == 0] = 0

density_score = density_vector.mean()
density_score
