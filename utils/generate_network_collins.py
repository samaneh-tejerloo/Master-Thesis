# -*- coding: utf-8 -*-
"""
Created on Sat May 10 09:30:08 2025

@author: tejer
"""


#%%
import pandas as pd
from goatools.anno.gaf_reader import GafReader
from gensim.models import KeyedVectors
import numpy as np
import json
import torch
from torch_geometric.data import Data
import networkx as nx
import torch_geometric
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
import matplotlib.pyplot as plt
import nocd
#%%
bp_embeddings = KeyedVectors.load_word2vec_format('datasets/biological_process_emb_128.txt')
cc_embeddings = KeyedVectors.load_word2vec_format('datasets/cellular_component_emb_128.txt')
mf_embeddings = KeyedVectors.load_word2vec_format('datasets/molecular_function_emb_128.txt')
#%%
df = pd.read_csv('datasets/colins.csv', index_col=0)
#%%
ogaf = GafReader('datasets/sgd.gaf')
#%%
with open('datasets/colins_semantic_name_to_sgd_id.json') as f:
    semantic_name_to_sgd_id = json.load(f)
#%%
ns2assc = ogaf.get_ns2assc()
#%%
def go_term_to_id(go_term):
    return str(int(go_term.split(':')[-1]))
#%%
def get_go_terms_and_embeddings(protein_semantic_name):
    try:
        mf_list = list(ns2assc['MF'][semantic_name_to_sgd_id[protein_semantic_name]])
        mf_emb = list(map(lambda x: mf_embeddings[go_term_to_id(x)], mf_list))
    except:
        mf_list = []
        mf_emb = []
    try:
        bp_list = list(ns2assc['BP'][semantic_name_to_sgd_id[protein_semantic_name]])
        bp_emb = list(map(lambda x: bp_embeddings[go_term_to_id(x)], bp_list))
    except:
        bp_list = []
        bp_emb = []
    # try:
    #     cc_list = list(ns2assc['CC'][semantic_name_to_sgd_id[protein_semantic_name]])
    #     cc_emb = list(map(lambda x: cc_embeddings[go_term_to_id(x)], cc_list))
    # except:
    #     cc_list = []
    #     cc_emb = []
        
    # go_terms = mf_list + bp_list + cc_list
    # go_embeddings = mf_emb + bp_emb + cc_emb
    go_terms = bp_list 
    go_embeddings = bp_emb
    return go_terms, go_embeddings

a,b = get_go_terms_and_embeddings('YOR089C')
#%%
proteins = sorted(list(set(df['protein1'].tolist() + df['protein2'].tolist())))
#%%
def protein_name_to_id(protein_name):
    return proteins.index(protein_name)

def id_to_protein_name(id):
    return proteins[id]
#%%
from tqdm import tqdm
edges_index = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    protein_id1 = protein_name_to_id(row['protein1'])
    protein_id2 = protein_name_to_id(row['protein2'])
    edges_index.append((protein_id1,protein_id2))
    edges_index.append((protein_id2,protein_id1))
#%%
# using go terms or embeddings
features = []
for protein in tqdm(proteins):
    go_terms, embeddings = get_go_terms_and_embeddings(protein)
    features.append(go_terms)
#%%
# only when using go terms
all_go_terms = []
for go_terms in features:
    for go_term in go_terms:
        all_go_terms.append(go_term)

all_go_terms = sorted(list(set(all_go_terms)))
print(len(all_go_terms))
#%%
# only when using go terms
one_hot_features= np.zeros((len(proteins), len(all_go_terms)))
#%%
# only when using go terms
for protein_idx, go_terms in enumerate(features):
    for go_term in go_terms:
        go_term_idx = all_go_terms.index(go_term)
        one_hot_features[protein_idx][go_term_idx] = 1
#%%
# When we are using embeddings
features = []
for protein in tqdm(proteins):
    go_terms, embeddings = get_go_terms_and_embeddings(protein)
    if len(embeddings) > 0:
        embeddings = np.array(embeddings).mean(axis=0)
        features.append(embeddings)
    else:
        features.append(np.array([]))
#%%
# When we are using embeddings
missing_features = []
for idx, feature in enumerate(features):
    if len(feature) == 0:
        missing_features.append(idx)
#%%
# When we are using embeddings
for missed_idx in missing_features:
    connected_features = []
    for edge in edges_index:
        if edge[0] == missed_idx and edge[1] not in missing_features:
            connected_features.append(features[edge[1]])
    features[missed_idx] = np.array(connected_features).mean(axis=0)
#%%
features = torch.tensor(one_hot_features,dtype=torch.float32)
#features = torch.tensor(features,dtype=torch.float32)
#%%
edge_index = torch.LongTensor(edges_index).T
#%%
data = Data(x=features, edge_index=edge_index)
#%%
data.num_nodes
#%%
data.is_directed()
#%%
data.num_features
#%%
g = torch_geometric.utils.to_networkx(data, to_undirected=True)
nx.draw(g, node_size=30, font_size=7)
#%%
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, 512)
        self.norm1 = torch.nn.LayerNorm(128)
        self.conv2 = GCNConv(512, 250)
        self.norm2 = torch.nn.LayerNorm(40)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        #x = self.norm1(x)
        x = F.relu(x)
        #x = F.dropout(x, p= 0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
#%%
A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.int8)
A[data.edge_index[0] , data.edge_index[1]] = 1
#%%
decoder = nocd.nn.BerpoDecoder(len(proteins), A.sum().item(), balance_loss=True)
#%%
def loss_fn(F,A):
    sim_matrix = F @ F.T
    edge_indices = torch.where(A==1)
    pos_values = sim_matrix[edge_indices[0], edge_indices[1]]
    pos_values = torch.log(1-torch.exp(-pos_values-1e-4))
    term_1 = ((sim_matrix * (1-A)).sum() / (1-A).sum())
    term_2 = - (pos_values).sum() / A.sum()
    loss =  term_1 + term_2
    if term_2.isnan() or loss.item() == float('inf'):
        print(sim_matrix)
        print(pos_values)
        print(f'term 1: {term_1}')
        print(f'term 2: {term_2}')
        return sim_matrix
    return loss
#%%
model = GCN()
#%%
from tqdm import trange
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)

losses = []
epochs = 5000

for i in trange(epochs, desc=' training phase', leave=True):
    model.train()
    optimizer.zero_grad()
    F_out = model(data)
    loss = decoder.loss_full(F_out, A.numpy())
    loss.backward()
    optimizer.step()
    model.eval()
    F_out = model(data)
    threshold = 0.5
    clustering = (F_out > threshold).to(torch.int8)
    print(clustering.sum(axis=0))
    print(f'[{i+1}/epochs] loss: {loss.item()}')
    losses.append(loss.item())
#%%
plt.plot(losses)
#%%
model.eval()

with torch.no_grad():
    F_out = model(data)
#%%
threshold = 0.5
clustering = (F_out > threshold).to(torch.int8)
print(clustering.sum(axis=0))
#%%
colors = [
    'red', 'green', 'blue', 'yellow', 'orange',
    'purple', 'pink', 'brown', 'cyan', 'magenta',
    'lime', 'olive', 'teal', 'navy', 'maroon',
    'gold', 'silver', 'gray', 'black', 'coral','aqua', 'aquamarine', 'azure', 'beige', 'bisque',
    'blanchedalmond', 'blueviolet', 'burlywood', 'cadetblue', 'chartreuse', 'coral', 'cornflowerblue', 'cornsilk', 'crimson',
    'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid',
    'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise'
]

from matplotlib.colors import to_rgb

rgb_colors = np.array([to_rgb(color) for color in colors])
#%%
node_colors = []
for node_assignment in clustering:
    i = torch.where(node_assignment==1)[0]
    if len(i) == 0:
        color = np.array(to_rgb('chocolate'))
        node_colors.append(color)
        continue

    color_node = rgb_colors[i].reshape(-1,3).mean(axis=0)
    node_colors.append(color_node)
#%%
g = torch_geometric.utils.to_networkx(data, to_undirected=True)
pos = nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos, node_size=30, node_color=node_colors)
nx.draw_networkx_edges(g, pos, alpha=0.1)
plt.show()
#%%
complexes = []
with open('datasets/sgd.txt','r') as f:
    for line in f.readlines():
        complexes.append(line.split())
#%%
# complexes that all of the proteins are in collins as well
filtered_complexes1 = []
for comp in complexes:
    flag = False
    for protein in comp:
        if protein not in proteins:
            flag=True
            break
    if not flag:
        filtered_complexes1.append(comp)
#%%
# complexes that contains at least two proteins from collins
filtered_complexes2 = []
for comp in complexes:
    counter = 0
    for protein in comp:
        if protein in proteins:
            counter+=1
    if counter >= 2:
        filtered_complexes2.append(comp)
#%%
# filters from complexes the proteins that only are present in collins
filtered_complexes3 = []
for comp in complexes:
    comp_new = []
    for protein in comp:
        if protein in proteins:
            comp_new.append(protein)
    if len(comp_new) > 0:
        filtered_complexes3.append(comp_new)
#%%
torch.save(model.state_dict(), 'results/nocd_250_d_bp_embed.pt')
#%%
# evaluating the model
#model.load_state_dict(torch.load('results/nocd_250_d.pt'))
model.eval()
F_out = model(data)
#%%
def calculate_metrics(F_out, threshold, reference_complex, threshold_na=0.25):
    clustering = (F_out > threshold).to(torch.int8)
    #print(clustering.sum(1))
    algorithm_complexes = []
    print('Finding the algorithm complexes...')
    for cluster_id in range(clustering.shape[1]):
        indices = torch.where(clustering[:,cluster_id]==1)[0]
        comp = list(np.array(proteins)[indices])
        if len(comp) != 0:
            algorithm_complexes.append(comp)
    
    min_algorithm_complexes = min(map(lambda x: len(x), algorithm_complexes))
    max_algorithm_complexes = max(map(lambda x: len(x), algorithm_complexes))
    print(f'min complex length found by algorithm is {min_algorithm_complexes}, and max: {max_algorithm_complexes}')
    
    print(f"Calculating the NCP and NCB with NA threshold: {threshold_na}...")
    
    NCP = 0
    NCB = 0
    
    algorithm_to_reference = {}
    for a_idx, a_comp in enumerate(algorithm_complexes):
        for idx, comp in enumerate(reference_complex):
            if len(a_comp) == 0:
                continue
            a_comp = set(a_comp)
            comp = set(comp)
            NA = len(a_comp.intersection(comp))**2 / (len(a_comp) * len(comp))
            if NA >= threshold_na:
                NCP+=1
                algorithm_to_reference[a_idx] = idx
                break
    
    reference_to_algorithm = {}
    for idx, comp in enumerate(reference_complex):
        for a_idx, a_comp in enumerate(algorithm_complexes):
            if len(a_comp) == 0:
                continue
            a_comp = set(a_comp)
            comp = set(comp)
            NA = len(a_comp.intersection(comp))**2 / (len(a_comp) * len(comp)) 
            if NA >= threshold_na:
                NCB+=1
                reference_to_algorithm[idx] = a_idx
                break
    print(f"NCP: {NCP}, NCB:{NCB}")
    recall = NCB/len(reference_complex)
    precision = NCP/len(algorithm_complexes)
    print('Recall:',recall)
    print('Precision:',precision)
    
    if recall + precision != 0:
        f = 2 * precision * recall / (precision + recall)
        print('F-Score:', f)
        return NCP,NCB, recall, precision, f
    else:
        return NCP,NCB, recall, precision, 0
#%%
threshold_list= np.arange(0,1,0.05)
recall_list = []
precision_list = []

for threshold in threshold_list:
    print('Calculating metrics for threshold:', threshold)
    _,_,recall, precision, _ = calculate_metrics(F_out, threshold, filtered_complexes3)
    print('#'*30)
    recall_list.append(recall)
    precision_list.append(precision)
#%%
recall_list = np.array(recall_list)
precision_list = np.array(precision_list)
#%%
sorted_indices = np.argsort(recall_list)
recall_list = recall_list[sorted_indices]
precision_list = precision_list[sorted_indices]
threshold_list = threshold_list[sorted_indices]

max_value = max(recall_list.max(), precision_list.max())

plt.plot(recall_list, precision_list,label='Precision-Recall Curve')
plt.plot(recall_list,recall_list, '--', label='x=y')
for idx, threshold in enumerate(threshold_list):
    plt.text(recall_list[idx],precision_list[idx],str(np.round(threshold,2)))
plt.scatter(recall_list, precision_list,color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim(max_value)
plt.ylin(max_value)
plt.legend()
plt.show()
#%%
