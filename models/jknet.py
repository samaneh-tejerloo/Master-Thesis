#%%
import torch
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import functional as F
import torch.nn as nn

class JKNetGATWith3Layers(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, heads=1, dropout=0):
        super().__init__()
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim

        self.gatconv1 = GATConv(embedding_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)
        self.gatconv2 = GATConv(intermediate_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)
        self.gatconv3 = GATConv(intermediate_dim, encoding_dim, heads=heads, concat=False, dropout=dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.gatconv1(x, edge_index)
        x1 = F.relu(x1)
        
        x2 = self.gatconv2(x1, edge_index)
        x2 = F.relu(x2)
        
        x3 = self.gatconv3(x2, edge_index)
        x3 = F.relu(x3)
        
        out = torch.vstack([x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)])
        out_pooled, _  = torch.max(out, dim=0)
        return out_pooled

class JKNetGATConcat(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, heads=1, dropout=0):
        super().__init__()
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim

        self.gatconv1 = GATConv(embedding_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)
        self.gatconv2 = GATConv(intermediate_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.gatconv1(x, edge_index)
        x1 = F.relu(x1)
        
        x2 = self.gatconv2(x1, edge_index)
        x2 = F.relu(x2)
        
        
        out = torch.cat([x1,x2], dim=-1)
        return out

class LSTMAttentionJKNETGAT(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, heads=1, dropout=0):
        super().__init__()
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim

        self.gatconv1 = GATConv(embedding_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)
        self.gatconv2 = GATConv(intermediate_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)

        self.lstm = nn.LSTM(input_size=intermediate_dim, hidden_size=intermediate_dim, batch_first=False, bidirectional=True)

        self.attention_project = nn.Linear(2 * intermediate_dim, 1, bias=False)

        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.gatconv1(x, edge_index)
        x1 = F.relu(x1)
        
        x2 = self.gatconv2(x1, edge_index)
        x2 = F.relu(x2)
        
        out = torch.vstack([x1.unsqueeze(0),x2.unsqueeze(0)])
        
        lstm_out, _ = self.lstm(out)

        A = F.softmax(self.attention_project(lstm_out), dim=0).squeeze().T

        out = torch.einsum('nk,knd->nd',A,out)
        return out

class AttentionJKNETGAT(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, heads=1, dropout=0, heads_transformer=4, transformer_layers=2):
        super().__init__()
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
        self.cls_token = nn.Parameter(torch.rand((1,intermediate_dim), requires_grad=True))

        self.gatconv1 = GATConv(embedding_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)
        self.gatconv2 = GATConv(intermediate_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)

        self.pos_embedding = nn.Embedding(3, intermediate_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(intermediate_dim, nhead=heads_transformer, dim_feedforward= 4 * intermediate_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=transformer_layers)

        self.norm = nn.LayerNorm(intermediate_dim)


        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.gatconv1(x, edge_index)
        x1 = F.relu(x1)
        
        x2 = self.gatconv2(x1, edge_index)
        x2 = F.relu(x2)
        
        out = torch.concat([x1.unsqueeze(1),x2.unsqueeze(1)], dim=1)
        cls_expanded = self.cls_token.unsqueeze(0).repeat(out.shape[0], 1, 1)
        out =torch.concat([cls_expanded, out], dim=1)
        out += self.pos_embedding(torch.arange(out.shape[1], device=out.device))

        out = self.transformer_encoder(out)
        out = self.norm(out)
        out = out[:,0,:]

        return out
#%%
if __name__ == '__main__':
    from dataset import PPIDataLoadingUtil
    data = PPIDataLoadingUtil('datasets/tadw-sc/krogan-core/krogan-core.csv', load_embeddings=False)
    features = torch.tensor(data.get_features('one_hot', name_spaces=['BP']), dtype=torch.float32)
    edge_index = torch.LongTensor(data.edges_index).T
    from torch_geometric.data import Data
    data = Data(x=features, edge_index=edge_index)
    net = LSTMAttentionJKNETGAT(embedding_dim=data.num_features, intermediate_dim=512, encoding_dim=512, heads=4)
    out = net(data)
    print(out.shape)
# %%
