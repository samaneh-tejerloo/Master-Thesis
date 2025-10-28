#%%
import torch
import torch_geometric.nn as gnn
from torch.nn import functional as F
import torch.nn as nn
from models.base import BaseGNN

class SimpleGNN(BaseGNN):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, n_layers=2, layer_module=gnn.GATConv, activation=F.relu, **kwargs):
        super().__init__(embedding_dim, intermediate_dim, encoding_dim, n_layers, layer_module, activation, **kwargs)
    
    def forward(self, data):
        outs = super().forward(data)
        return outs[-1]

if __name__ == '__main__':
    from dataset import PPIDataLoadingUtil
    from torch_geometric.data import Data
    data = PPIDataLoadingUtil('datasets/tadw-sc/collins_2007/colins2007.csv', load_embeddings=False)

    features = data.get_features('one_hot', name_spaces=['BP'])
    features = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(data.edges_index).long().T
    data = Data(features, edge_index)

    model = SimpleGNN(embedding_dim=data.num_features, intermediate_dim=256, encoding_dim=256, n_layers=3, layer_module=gnn.GATConv, activation=F.relu, heads=4, dropout=0)

    out = model(data)
    print(out.shape)
# %%
