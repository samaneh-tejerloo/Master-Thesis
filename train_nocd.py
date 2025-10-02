#%%
from dataset import PPIDataLoadingUtil
from models import SimpleGCN
from torch_geometric.data import Data
import torch
import nocd
from tqdm import tqdm
from evaluate import Evaluation
from constants import SGD_GOLD_STANDARD_PATH
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# %%
ppi_data_loader = PPIDataLoadingUtil('datasets/tadw-sc/collins_2007/colins2007.csv')
# %%
features = ppi_data_loader.get_features(type='one_hot', name_spaces=['BP'])
features = torch.tensor(features, dtype=torch.float32)
edge_index = torch.LongTensor(ppi_data_loader.edges_index).T
#%%
data = Data(x=features, edge_index=edge_index)
# %%
model = SimpleGCN(data.num_features, 512, 256)
decoder = nocd.nn.BerpoDecoder(data.num_nodes, data.num_edges, balance_loss=False)
#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.int8)
A[data.edge_index[0] , data.edge_index[1]] = 1
#%%
epochs = 1000
model.train()
# progress_bar = tqdm(range(epochs))
for epoch in range(epochs):
    optimizer.zero_grad()
    F_out = model(data)
    loss = decoder.loss_full(F_out, A.numpy())
    loss.backward()
    optimizer.step()
    # progress_bar.set_description(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
    print(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
#%%
model.eval()
with torch.no_grad():
    F_out = model(data)
evaluator = Evaluation('datasets/golden standard/ada_ppi.txt', ppi_data_loader)
evaluator.filter_reference_complex(filtering_method='all_proteins_in_dataset')

for threshold in np.arange(0.1,1,0.1):
    threshold = np.round(threshold,1).item()
    print(f'threshold = {threshold}')
    # threshold = 0.5
    clustering = (F_out > threshold).to(torch.int8)
    # print(clustering.sum(dim=0))

    algorithm_complexes = []
    for cluser_id in range(clustering.shape[1]):
        indices = torch.where(clustering[:, cluser_id] ==1)[0]
        if len(indices) > 0:
            alg_complex = []
            for protein_idx in indices.tolist():
                protein_name = ppi_data_loader.id_to_protein_name(protein_idx)
                alg_complex.append(protein_name)
            algorithm_complexes.append(alg_complex)

    print('Number of clusters', len(algorithm_complexes))
    print('Number of clusters with one protein', sum([len(c) <= 1 for c in algorithm_complexes]))
    algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
    print('Number of algorithm complexes:', len(algorithm_complexes))
    # evaluator.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
    result = evaluator.evalute(algorithm_complexes)
    print(result)
    print('#'*100)
# %%
#%%
torch.save(model.state_dict(), 'checkpoints/nocd_256_64_5k_epoch/model.pt')
# %%
tsne = TSNE(n_components=2)
nocd_tsne_embeddings = tsne.fit_transform(F_out)
# %%
plt.scatter(nocd_tsne_embeddings[:,0], nocd_tsne_embeddings[:,1], s=1)
#%%
threshold = 0.3
clustering = (F_out > threshold).to(torch.int8)
# print(clustering.sum(dim=0))

algorithm_complexes = []
for cluser_id in range(clustering.shape[1]):
    indices = torch.where(clustering[:, cluser_id] ==1)[0]
    if len(indices) > 0:
        alg_complex = []
        for protein_idx in indices.tolist():
            protein_name = ppi_data_loader.id_to_protein_name(protein_idx)
            alg_complex.append(protein_name)
        algorithm_complexes.append(alg_complex)

print('Number of clusters', len(algorithm_complexes))
print('Number of clusters with one protein', sum([len(c) <= 1 for c in algorithm_complexes]))
algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
print('Number of algorithm complexes:', len(algorithm_complexes))
# evaluator.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
result = evaluator.evalute(algorithm_complexes)
print(result)
print('#'*100)
# %%
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(min_samples=2, eps=0.5, metric='cosine').fit(F_out)
dbscan_clusters = dbscan.labels_

dbscan_algorithm_complexes = []
for cluser_id in range(dbscan_clusters.max()):
    indices = np.where(dbscan_clusters == 0)[0]
    if len(indices) > 0:
        alg_complex = []
        for protein_idx in indices.tolist():
            protein_name = ppi_data_loader.id_to_protein_name(protein_idx)
            alg_complex.append(protein_name)
        dbscan_algorithm_complexes.append(alg_complex)

print('Number of clusters', len(dbscan_algorithm_complexes))
print('Number of clusters with one protein', sum([len(c) <= 1 for c in dbscan_algorithm_complexes]))
dbscan_algorithm_complexes = [c for c in dbscan_algorithm_complexes if len(c) > 1]
print('Number of algorithm complexes:', len(dbscan_algorithm_complexes))
# evaluator.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
result = evaluator.evalute(dbscan_algorithm_complexes)
print(result)
#%%
evaluator.evalute(algorithm_complexes + dbscan_algorithm_complexes)
# %%
