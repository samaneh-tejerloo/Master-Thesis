#%%
import networkx as nx
import numpy as np
import torch

def get_clique(adj):
    edge = [(n1, n2) for n1, n2 in zip(adj.nonzero()[0], adj.nonzero()[1])]

    G = nx.Graph()
    G.add_edges_from(edge)

    cliques = nx.find_cliques(G)

    return [c for c in cliques if len(c)>=3]
#%%
sim = torch.load('a.pt')
#%%
# %%
cliques = get_clique(sim)
# %%
