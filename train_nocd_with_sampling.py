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
#%%
BALANCE = True
Weighted = False
DATASET_PATH = 'datasets/AdaPPI/Krogan-core/krogan2006core.csv'
IS_ADA_PPI = True
EPOCHS = 2000
LAM = 1
# only affects the TADW_SC datasets
NAME_SPACES = ['BP', 'MF']
# %%
ppi_data_loader = PPIDataLoadingUtil(DATASET_PATH,load_embeddings=False, ada_ppi_dataset=IS_ADA_PPI, load_weights=Weighted)
# %%
features = ppi_data_loader.get_features(type='one_hot', name_spaces=NAME_SPACES)
features = torch.tensor(features, dtype=torch.float32)
edge_index = torch.LongTensor(ppi_data_loader.edges_index).T
# %%
data = Data(x=features, edge_index=edge_index)
# %%
import torch
from torch_geometric.nn import GCNConv, GAT
from torch.nn import functional as F

class SimpleGCN(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=512, encoding_dim=250):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim

        self.conv1 = GCNConv(embedding_dim, intermediate_dim)
        self.conv2 = GCNConv(intermediate_dim, encoding_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PairNorm(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        denom = (x.pow(2).sum(dim=1).mean().sqrt() + 1e-12)
        return self.scale * x / denom

class BetterGCN(nn.Module):
    def __init__(
        self,
        in_dim,
        h1=512,
        h2=256,
        dropout=0.4,
        use_pairnorm=True,
        use_residual=True,
    ):
        super().__init__()
        self.use_pairnorm = use_pairnorm
        self.use_residual = use_residual

        self.conv1 = GCNConv(in_dim, h1, cached=False, add_self_loops=True, normalize=True)
        self.bn1   = nn.BatchNorm1d(h1)
        self.conv2 = GCNConv(h1, h2, cached=False, add_self_loops=True, normalize=True)
        self.bn2   = nn.BatchNorm1d(h2)

        self.drop = nn.Dropout(dropout)
        self.pn1  = PairNorm()
        self.pn2  = PairNorm()

        # If using residual and dims differ, add a projection
        self.proj1 = nn.Linear(in_dim, h1) if use_residual and in_dim != h1 else None
        self.proj2 = nn.Linear(h1, h2)     if use_residual and h1 != h2 else None

    def forward(self, data):
        x, ei = data.x, data.edge_index

        # Layer 1
        h1 = self.conv1(x, ei)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.drop(h1)
        if self.use_pairnorm: h1 = self.pn1(h1)
        if self.use_residual:
            r1 = x if self.proj1 is None else self.proj1(x)
            h1 = h1 + r1

        # Layer 2
        h2 = self.conv2(h1, ei)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)            # keep ReLU if using BerPo (>0 dot-products)
        h2 = self.drop(h2)
        if self.use_pairnorm: h2 = self.pn2(h2)
        if self.use_residual:
            r2 = h1 if self.proj2 is None else self.proj2(h1)
            h2 = h2 + r2

        return h2  # e.g., 256-D
#%%
import math
import torch
from torch_geometric.utils import negative_sampling

class EdgeBatcher:
    def __init__(self, pos_edge_index, num_nodes, batch_size=1500, balance=True, weighted=True):
        self.pos = pos_edge_index
        if weighted:
            self.weights = torch.tensor(ppi_data_loader.weights)
        else:
            self.weights = None
        self.N = num_nodes
        self.S = batch_size
        self.balance = balance
        self.num_pos = pos_edge_index.size(1)

    def __iter__(self):
        perm = torch.randperm(self.num_pos)
        # walk the positive edges in chunks of S
        for start in range(0, self.num_pos, self.S):
            end = min(start + self.S, self.num_pos)
            pos_batch = self.pos[:, perm[start:end]]

            # sample exactly the same number of negatives (balanced)
            if self.balance:
                num_neg = pos_batch.size(1)
            else:
                # you can choose a ratio, e.g., 3x negatives
                num_neg = 3 * pos_batch.size(1)

            neg_batch = negative_sampling(
                edge_index=self.pos,
                num_nodes=self.N,
                num_neg_samples=num_neg,
                method='sparse',
                force_undirected=True
            )
            if self.weights is not None:
                weights = self.weights[perm[start:end]]
            else:
                weights = None
            yield pos_batch, neg_batch, weights
        
    def __len__(self):
        """Return the number of mini-batches per epoch."""
        return math.ceil(self.num_pos / self.S)
#%%
dataloader = EdgeBatcher(pos_edge_index=data.edge_index, num_nodes=data.num_nodes, batch_size=1500, balance=True, weighted=Weighted)
# %%
# model = SimpleGCN(data.num_features, 512, 256)
model = BetterGCN(data.num_features, 512, 256, 0.4, True, True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float32)
if Weighted:
    A[data.edge_index[0] , data.edge_index[1]] = torch.tensor(ppi_data_loader.weights, dtype=torch.float32)
else:
    A[data.edge_index[0], data.edge_index[1]] = 1
# %%
decoder = BerpoDecoder(data.num_nodes, A.sum().item(), balance_loss=BALANCE)
# %%
epochs = EPOCHS
model.train()
# progress_bar = tqdm(range(epochs))
for epoch in range(epochs):
    running_loss = 0
    for b_pos, b_neg, b_pos_weights in dataloader:
        b_pos = b_pos.T
        b_neg = b_neg.T
        F_out = model(data)
        opt.zero_grad()
        loss = decoder.loss_batch(F_out, b_pos, b_neg, pos_weights=b_pos_weights)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    # progress_bar.set_description(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
    print(f'Epoch: {epoch+1:02}/{epochs}, loss:{running_loss / len(dataloader):.4f}')
# %%
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
#%%