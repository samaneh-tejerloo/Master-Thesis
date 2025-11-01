#%%
import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F

class BaseGNN(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, n_layers=2, layer_module=gnn.GATConv, activation= F.relu,**kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.encoding_dim = encoding_dim
        self.n_layers = n_layers
        self.layer_module = layer_module
        self.activation = activation
        
        layers = []
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                d1 = self.embedding_dim
                d2 = self.intermediate_dim
            elif layer_idx == n_layers - 1:
                d1 = self.intermediate_dim
                d2 = self.encoding_dim
            else:
                d1 = self.intermediate_dim
                d2 = self.intermediate_dim
            
            if layer_module == gnn.GATConv or layer_module== gnn.GATv2Conv:
                heads = kwargs.get('heads', 4)
                concat = kwargs.get('concat', False)
                dropout = kwargs.get('dropout', 0)
                residual = kwargs.get('res', False)
                layer = layer_module(d1, d2, heads=heads, concat=concat, dropout=dropout, residual=residual)
            elif layer_module == gnn.GCNConv:
                layer = layer_module(d1, d2, add_self_loops=True)
            else:
                layer = layer_module(d1,d2)

            layers.append(layer)
        
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        outs = [x]
        for layer in self.layers:
            out = layer(outs[-1], edge_index)
            out = self.activation(out)
            outs.append(out)
        return outs[1:]

if __name__ == '__main__':
    from dataset import PPIDataLoadingUtil
    from torch_geometric.data import Data
    data = PPIDataLoadingUtil('datasets/tadw-sc/collins_2007/colins2007.csv', load_embeddings=False)

    features = data.get_features('one_hot', name_spaces=['BP'])
    features = torch.tensor(features, dtype=torch.float32)
    edge_index = torch.tensor(data.edges_index).long().T
    data = Data(features, edge_index)

    model = BaseGNN(embedding_dim=data.num_features, intermediate_dim=256, encoding_dim=256, n_layers=3, layer_module=gnn.GATConv, activation=F.relu, heads=4, dropout=0)

    out = model(data)
# %%
