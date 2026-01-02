#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 06:56:45 2026

@author: user
"""

import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
#%%
biogrid_path = "logs/history/metric_biogrid_SimpleGNN_GAT_2-layers_4-heads_relu_BP_MF_512_5000.json"
krogan_path = "logs/history/metric_krogan_core_SimpleGNN_GAT_2_layers_4_heads_relu_BP_MF_512.json"
with open(biogrid_path) as f:
    biogrid_history = json.load(f)

with open(krogan_path) as f:
    krogan_history = json.load(f)
#%%
history = krogan_history

scaler = MinMaxScaler()
f1_normalized = scaler.fit_transform(np.array(history['F1']).reshape(-1,1))
modularity_normalized = scaler.fit_transform(np.array(history['modularity']).reshape(-1,1))
density_normalized = scaler.fit_transform(np.array(history["density"]).reshape(-1,1))
diff_normalized = scaler.fit_transform(np.array(history["diff"]).reshape(-1,1))
homo = np.array(history["homogenity"]).reshape(-1,1)
homo[homo == float('inf')] = 0
homo_normalized = scaler.fit_transform(homo)
sill = np.array(history["silhouette"]).reshape(-1,1)
sill[sill == float('inf')] = 0
sill_normalized = scaler.fit_transform(sill)

plt.figure()
max_f1 = f1_normalized.max()
argmax_f1 = f1_normalized.argmax()
plt.plot(f1_normalized, label="F1")
plt.scatter([argmax_f1], [max_f1], label="max F1", color='red', zorder=20, alpha=0.3)
plt.plot(modularity_normalized, label="modularity")
plt.plot(density_normalized, label="density")
plt.plot(diff_normalized, label="diff", color='purple')
plt.plot(homo_normalized, label="diff", color='black')
plt.plot(sill_normalized, label="diff", color='pink')
plt.legend()
plt.title("korogan" if len(history['F1']) == 2000 else "biogrid")
plt.show()
#%%
from dataset import PPIDataLoadingUtil
from evaluate import Evaluation
#%%
dataset = PPIDataLoadingUtil("datasets/tadw-sc/krogan-core/krogan-core.csv", load_embeddings=False, load_weights=True)
#%%
evaluator = Evaluation("datasets/golden standard/ada_ppi.txt",dataset)
evaluator.filter_reference_complex("just_keep_dataset_proteins")
#%%
reference_complexes = evaluator.filtered_complexes
reference_complexes_ids = [[dataset.protein_name_to_id(p) for p in protein_complex] for protein_complex in reference_complexes]
#%%
features = dataset.get_features('one_hot', name_spaces=['MF','BP'])
#%%
sum_commom_features = 0
for protein_complex in reference_complexes_ids:
    features_complex = features[protein_complex]
    sum_commom_features += int(np.all(features_complex, axis=0).sum())
metric = sum_commom_features / len(reference_complexes_ids)
print(metric)
#%%
from models import SimpleGNN
import torch_geometric.nn as gnn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from itertools import chain
model = SimpleGNN(
    embedding_dim=features.shape[1],
    intermediate_dim=512,
    encoding_dim=512,
    n_layers=2,
    layer_module=gnn.GATConv,
    activation=F.relu,
    heads=4,
)
model.load_state_dict(torch.load("logs/weights/metric_krogan_core_SimpleGNN_GAT_2_layers_4_heads_relu_BP_MF_512.pt"))
#%%
model.eval()
edge_index = torch.LongTensor(dataset.edges_index).T
data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index)
with torch.no_grad():
    F_out = model(data)
#%%
threshold=0.3
clustering = (F_out > threshold).to(torch.int8)

algorithm_complexes = []
for cluser_id in range(clustering.shape[1]):
    indices = torch.where(clustering[:, cluser_id] == 1)[0]
    if len(indices) > 0:
        alg_complex = []
        for protein_idx in indices.tolist():
            protein_name = dataset.id_to_protein_name(protein_idx)
            alg_complex.append(protein_name)
        algorithm_complexes.append(alg_complex)
#%%
algorithm_complexes_ids = [[dataset.protein_name_to_id(p) for p in protein_complex] for protein_complex in algorithm_complexes]
#%%
sum_commom_features = 0
for protein_complex in algorithm_complexes_ids:
    features_complex = features[protein_complex]
    sum_commom_features += int(np.all(features_complex, axis=0).sum())
metric = sum_commom_features / len(algorithm_complexes_ids)
print(metric)
#%%
len(reference_complexes)
#%%
len(algorithm_complexes)
#%%
def get_algorithm_complexes(F_out, threshold=0.3):
    clustering = (F_out > threshold).to(torch.int8)

    algorithm_complexes = []
    for cluser_id in range(clustering.shape[1]):
        indices = torch.where(clustering[:, cluser_id] == 1)[0]
        if len(indices) > 0:
            alg_complex = []
            for protein_idx in indices.tolist():
                protein_name = dataset.id_to_protein_name(protein_idx)
                alg_complex.append(protein_name)
            algorithm_complexes.append(alg_complex)

    print("Number of clusters", len(algorithm_complexes))
    print(
        "Number of clusters with one protein",
        sum([len(c) <= 1 for c in algorithm_complexes]),
    )
    algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
    print("Number of algorithm complexes:", len(algorithm_complexes))
    return algorithm_complexes
def merge_unique(lists):
    all_groups = chain(*lists)
    uniq = set(frozenset(group) for group in all_groups)
    return [sorted(list(g)) for g in uniq]
#%%
algorithm_complexes_2 = get_algorithm_complexes(F_out, threshold=0.2)
algorithm_complexes_3 = get_algorithm_complexes(F_out, threshold=0.3)
algorithm_complexes_4 = get_algorithm_complexes(F_out, threshold=0.4)
algorithm_complexes_5 = get_algorithm_complexes(F_out, threshold=0.5)
algorithm_complexes_6 = get_algorithm_complexes(F_out, threshold=0.6)
algorithm_complexes_7 = get_algorithm_complexes(F_out, threshold=0.7)

all_complexes = [
    algorithm_complexes_2,
    algorithm_complexes_3,
    algorithm_complexes_4,
    algorithm_complexes_5,
    algorithm_complexes_6,
    algorithm_complexes_7,
]
complexes = merge_unique(all_complexes)
print(len(complexes))
complexes = [p for p in complexes if len(p) > 2]
print(len(complexes))
result = evaluator.evalute(complexes)
print(result)
#%%
algorithm_complexes_ids = [[dataset.protein_name_to_id(p) for p in protein_complex] for protein_complex in complexes]
sum_commom_features = 0
for protein_complex in algorithm_complexes_ids:
    features_complex = features[protein_complex]
    sum_commom_features += int(np.all(features_complex, axis=0).sum())
metric = sum_commom_features / len(algorithm_complexes_ids)
print(metric)