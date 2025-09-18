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
#%%
ppi_data_loader = PPIDataLoadingUtil('datasets/colins.csv')
# %%
features = np.eye(len(ppi_data_loader.proteins))
features = torch.tensor(features, dtype=torch.float32)
edge_index = torch.LongTensor(ppi_data_loader.edges_index).T

data = Data(x=features, edge_index=edge_index)
# %%
model = SimpleGCN(data.num_features, 256, 256)
decoder = nocd.nn.BerpoDecoder(data.num_nodes, data.num_edges, balance_loss=False)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.int8)
A[data.edge_index[0] , data.edge_index[1]] = 1

epochs = 1000
model.train()
progress_bar = tqdm(range(epochs))
for epoch in progress_bar:
    optimizer.zero_grad()
    F_out = model(data)
    loss = decoder.loss_full(F_out, A.numpy())
    loss.backward()
    optimizer.step()
    progress_bar.set_description(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
#%%
model.eval()
with torch.no_grad():
    F_out = model(data)
#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
tsne_embeddings = tsne.fit_transform(F_out.numpy())
#%%
import matplotlib.pyplot as plt
from evaluate import Evaluation
evaluate = Evaluation('datasets/sgd.txt', ppi_data_loader)
evaluate.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
complexes = evaluate.filtered_complexes

for idx, complex in enumerate(complexes):
    plt.figure(figsize=(10,10))
    colors = ['green' if protein in complex else 'yellow' for protein in ppi_data_loader.proteins]

    plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], s=5, c=colors, alpha=0.5)

    plt.savefig(f'temp/gcn_two_layer_structural/{idx}.jpg')
# %%
