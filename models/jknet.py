#%%
import torch
import torch_geometric.nn as gnn
from torch.nn import functional as F
import torch.nn as nn
from models.base import BaseGNN
from typing import List


class JKNetMaxPooling(BaseGNN):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, n_layers=2, layer_module=gnn.GATConv, activation=F.relu, **kwargs):
        super().__init__(embedding_dim, intermediate_dim, encoding_dim, n_layers, layer_module, activation, **kwargs)
    
    def forward(self, data):
        outs: List[torch.Tensor] = super().forward(data)
        outs = torch.vstack([out.unsqueeze(0) for out in outs])

        out_pooled, _ = torch.max(outs, dim=0)
        return out_pooled

class JKNetConcat(BaseGNN):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, n_layers=2, layer_module=gnn.GATConv, activation=F.relu, **kwargs):

        super().__init__(embedding_dim, intermediate_dim, encoding_dim, n_layers, layer_module, activation, **kwargs)

    def forward(self, data):
        outs:List[torch.Tensor] = super().forward(data)
        out = torch.concat(outs, dim=-1)
        return out

class JKNetLSTMAttention(BaseGNN):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, n_layers=2, layer_module=gnn.GATConv, activation=F.relu, **kwargs):
        super().__init__(embedding_dim, intermediate_dim, encoding_dim, n_layers, layer_module, activation, **kwargs)

        bidirectional = kwargs.get('bidirectional',True)

        self.lstm = nn.LSTM(input_size=encoding_dim, hidden_size=encoding_dim, batch_first=False, bidirectional=bidirectional)

        coef_dim = 2 if bidirectional else 1
        self.attention_proj = nn.Linear(coef_dim * encoding_dim, 1, bias=False)
    
    def forward(self, data):
        outs: List[torch.Tensor] = super().forward(data)
        outs = torch.concat([out.unsqueeze(0) for out in outs], dim=0)
        lstm_out, _ = self.lstm(outs)
        A_hat = self.attention_proj(lstm_out).squeeze()
        A = F.softmax(A_hat, dim=0).T

        out = torch.einsum('nk,knd->nd', A, outs)
        return out

class JKNetMultiHeadAttention(BaseGNN):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, n_layers=2, layer_module=gnn.GATConv, activation=F.relu, **kwargs):

        super().__init__(embedding_dim, intermediate_dim, encoding_dim, n_layers, 
        layer_module, activation, **kwargs)

        self.cls_token = nn.Parameter(torch.rand(1,encoding_dim), requires_grad=True)

        self.pos_embeddings = nn.Embedding(n_layers + 1, embedding_dim=encoding_dim)

        transformer_heads = kwargs.get('transformer_heads', 4)

        transformer_layers = kwargs.get('transformer_layers', 1)

        mlp_ratio = kwargs.get('mlp_ratio', 4)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(encoding_dim, nhead=transformer_heads, dim_feedforward=mlp_ratio * encoding_dim, batch_first=False)

        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=transformer_layers)
    
    def forward(self, data):
        outs: List[torch.Tensor] = super().forward(data)
        
        outs = torch.concat([out.unsqueeze(0) for out in outs], dim=0)
        cls_expanded = self.cls_token.unsqueeze(1).expand(1,outs.shape[1], outs.shape[2])
        outs = torch.concat([cls_expanded, outs], dim=0)
        print(outs.shape)
        outs += self.pos_embeddings(torch.arange(outs.shape[0], device=outs.device)).unsqueeze(1)

        outs = self.transformer_encoder(outs)
        return outs[0,:,:]


if __name__ == '__main__':
    from dataset import PPIDataLoadingUtil
    data = PPIDataLoadingUtil('datasets/tadw-sc/krogan-core/krogan-core.csv', load_embeddings=False)
    features = torch.tensor(data.get_features('one_hot', name_spaces=['BP']), dtype=torch.float32)
    edge_index = torch.LongTensor(data.edges_index).T
    from torch_geometric.data import Data
    data = Data(x=features, edge_index=edge_index)
    net = JKNetMultiHeadAttention(embedding_dim=data.num_features, intermediate_dim=512, encoding_dim=512, heads=4, n_layers=3, layer_module=gnn.GATConv)
    out = net(data)
    print(out.shape)
# %%