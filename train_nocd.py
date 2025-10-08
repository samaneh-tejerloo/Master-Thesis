#%%
from dataset import PPIDataLoadingUtil
from models import SimpleGCN
from torch_geometric.data import Data
import torch
from nocd_decoder import BerpoDecoder
from tqdm import tqdm
from evaluate import Evaluation
from constants import SGD_GOLD_STANDARD_PATH
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
# %%
BALANCE = False
Weighted = True
DATASET_PATH = 'datasets/AdaPPI/DIP/dip.csv'
IS_ADA_PPI = False
EPOCHS = 2000
LAM = 1
# only affects the TADW_SC datasets
NAME_SPACES = ['BP', 'MF']
#%%
ppi_data_loader = PPIDataLoadingUtil(DATASET_PATH,load_embeddings=False, ada_ppi_dataset=IS_ADA_PPI, load_weights=Weighted)
# %%
features = ppi_data_loader.get_features(type='one_hot', name_spaces=NAME_SPACES)
features = torch.tensor(features, dtype=torch.float32)
edge_index = torch.LongTensor(ppi_data_loader.edges_index).T
#%%
data = Data(x=features, edge_index=edge_index)
# %%
model = SimpleGCN(data.num_features, 512, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float32)
if Weighted:
    A[data.edge_index[0] , data.edge_index[1]] = torch.tensor(ppi_data_loader.weights, dtype=torch.float32)
else:
    A[data.edge_index[0], data.edge_index[1]] = 1

A_2 = A @ A
A_2[A_2 > 0] = 1
for i in range(A.shape[0]):
    A_2[i,i] = 0
#%%
# decoder = nocd.nn.BerpoDecoder(data.num_nodes, data.num_edges, balance_loss=BALANCE)
decoder = BerpoDecoder(data.num_nodes, A.sum().item(), balance_loss=BALANCE)
decoder_2 = BerpoDecoder(data.num_nodes, A_2.sum().item(), balance_loss=BALANCE)
#%%
epochs = EPOCHS
model.train()
# progress_bar = tqdm(range(epochs))
for epoch in range(epochs):
    optimizer.zero_grad()
    F_out = model(data)
    if Weighted:
        if LAM == 0:
            loss_1 = decoder.loss_full_weighted(F_out, A)
            loss = loss_1
        elif LAM == 1:
            loss_2 = decoder_2.loss_full_weighted(F_out, A_2)
            loss = loss_2
        else:
            loss_1 = decoder.loss_full_weighted(F_out, A)
            loss_2 = decoder_2.loss_full_weighted(F_out, A_2)
            loss = (1-LAM) * loss_1 + (LAM) * loss_2
    else:
        if LAM == 0:
            loss_1 = decoder.loss_full(F_out, A.numpy())
            loss = loss_1
        elif LAM == 1:
            loss_2 = decoder_2.loss_full(F_out, A_2.numpy())
            loss = loss_2
        else:
            loss_1 = decoder.loss_full(F_out, A.numpy())
            loss_2 = decoder_2.loss_full(F_out, A_2.numpy())
            loss = (1-LAM) * loss_1 + (LAM) * loss_2
    loss.backward()
    optimizer.step()
    # progress_bar.set_description(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
    print(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
#%%
model.eval()
with torch.no_grad():
    F_out = model(data)
evaluator = Evaluation('datasets/golden standard/ada_ppi.txt', ppi_data_loader)
# evaluator.filter_reference_complex(filtering_method='all_proteins_in_dataset')
evaluator.filter_reference_complex(filtering_method='just_keep_dataset_proteins')

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
# torch.save(model.state_dict(), 'checkpoints/nocd_256_64_5k_epoch/model.pt')
# %%
# tsne = TSNE(n_components=2)
# nocd_tsne_embeddings = tsne.fit_transform(F_out)
# # %%
# plt.scatter(nocd_tsne_embeddings[:,0], nocd_tsne_embeddings[:,1], s=1)
#%%
threshold = 0.2
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
dbscan = DBSCAN(min_samples=7, eps=0.01, metric='cosine').fit(F_out)
dbscan_clusters = dbscan.labels_

dbscan_algorithm_complexes = []
for cluster_id in range(dbscan_clusters.max()):
    indices = np.where(dbscan_clusters == cluster_id)[0]
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
#%%
evaluator.evalute(algorithm_complexes)