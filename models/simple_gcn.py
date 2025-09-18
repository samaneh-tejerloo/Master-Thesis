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


class SimpleGAT(torch.nn.Module):
    def __init__(self, embedding_dim=128, intermediate_dim=256, encoding_dim=128):
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