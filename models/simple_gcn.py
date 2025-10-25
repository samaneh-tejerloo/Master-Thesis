import torch
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import functional as F
import torch.nn as nn

class SimpleGCN(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=512, encoding_dim=250, proj=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim

        self.conv1 = GCNConv(embedding_dim, intermediate_dim)
        self.conv2 = GCNConv(intermediate_dim, encoding_dim)
        if proj is not None:
            self.proj = nn.Linear(proj, embedding_dim)
        else:
            self.proj = None
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.proj:
            x = self.proj(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class SimpleGAT(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128, heads=1, dropout=0):
        super().__init__()
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim

        self.gatconv1 = GATConv(embedding_dim, intermediate_dim, heads=heads, concat=False, dropout=dropout)
        self.gatconv2 = GATConv(intermediate_dim, encoding_dim, heads=heads, concat=False, dropout=dropout)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gatconv1(x, edge_index)
        x = F.relu(x)
        x = self.gatconv2(x, edge_index)
        x = F.relu(x)
        return x