#%%
from dataset import PPIDataLoadingUtil
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch_geometric.nn as gnn
from nocd_decoder import BerpoDecoder
from evaluate import Evaluation
import json
import os
# %%
ppi_data_loader = PPIDataLoadingUtil('datasets/tadw-sc/krogan-core/krogan-core.csv', load_embeddings=True, load_weights=True)
# %%
bp_embeddings = ppi_data_loader.get_features('embedding', ['BP'])
mf_embeddings = ppi_data_loader.get_features('embedding', ['MF'])
cc_embeddings = ppi_data_loader.get_features('embedding', ['CC'])
num_nodes = len(bp_embeddings)
epochs = 100
base_dir='logs'
file_name=  'attention_feature_pooling_gat'
# %%
# max go terms in bp_embeddings
def process_embeddings(embeddings, embedding_dim):
    max_len = max([len(p) for p in embeddings])
    inputs = torch.zeros(len(embeddings), max_len, embedding_dim)
    pad_mask = torch.ones(len(embeddings), max_len).bool()
    for p_idx, p_features in enumerate(embeddings):
        for f_idx, feature in enumerate(p_features):
            inputs[p_idx,f_idx] = torch.tensor(feature)
            pad_mask[p_idx, f_idx] = 0
    
    return inputs, pad_mask

bp_inputs, bp_mask = process_embeddings(bp_embeddings, embedding_dim=128)
mf_inputs, mf_mask = process_embeddings(mf_embeddings, embedding_dim=128)
cc_inputs, cc_mask = process_embeddings(cc_embeddings, embedding_dim=128)
# %%
class FeatureAttentionPooling(nn.Module):
    def __init__(self, embedding_dim= 128, nhead=4, num_layers=1):
        super().__init__()
        self.attention_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=embedding_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.attention_layer, num_layers=1)
        self.cls = nn.Parameter(torch.tensor(128, dtype=torch.float32))
    
    def forward(self, inputs, pad_mask):
        N,S,D = inputs.shape
        inputs = torch.concat([self.cls.expand(N,1,D), inputs], dim=1)
        pad_mask = torch.concat([torch.zeros(N,1), pad_mask], dim=1)
        out = self.transformer(inputs, src_key_padding_mask=pad_mask)
        return out[:,0,:]
# %%
class FeatureAttentionGAT(nn.Module):
    def __init__(self, embedding_dim=128, nhead=4, attn_num_layers=1, gat_num_layers=2, intermediate_dim=512, encodding_dim=512):
        super().__init__()
        self.bp_feature_pooling = FeatureAttentionPooling(embedding_dim, nhead, attn_num_layers)
        self.mf_feature_pooling = FeatureAttentionPooling(embedding_dim, nhead, attn_num_layers)
        self.cc_feature_pooling = FeatureAttentionPooling(embedding_dim, nhead, attn_num_layers)

        self.gat = gnn.GAT(3*embedding_dim, intermediate_dim, gat_num_layers, encodding_dim)
    
    def forward(self, bp_inputs, bp_mask, mf_inputs, mf_mask, cc_input, cc_mask, edge_index):
        out_bp = self.bp_feature_pooling(bp_inputs, bp_mask)
        out_mf = self.bp_feature_pooling(mf_inputs, mf_mask)
        out_cc = self.cc_feature_pooling(cc_inputs, cc_mask)

        features = torch.concat([out_bp, out_mf, out_cc], dim=-1)

        out_gat = self.gat(features, edge_index)
        return out_gat
# %%
edge_index = torch.tensor(ppi_data_loader.edges_index).long().T
# %%
model = FeatureAttentionGAT()
#%%
def evaluate_model(model, evaluator, bp_inputs, bp_mask, mf_inputs, mf_mask, cc_inputs, cc_mask,edge_index, ppi_data_loader, do_print=False):
    # evaluating the model
    model.eval()
    with torch.no_grad():
        F_out = model(bp_inputs, bp_mask, mf_inputs, mf_mask, cc_inputs, cc_mask, edge_index)

    max_f1 = -1
    best_threshold = -1
    best_result = {
                'Precision': -1,
                'Recall': -1,
                'Acc': -1,
                'F1': -1,
                'NCP':-1,
                'NCB': -1,
            }
    for threshold in np.arange(0.1,1,0.1):
        threshold = np.round(threshold,1).item()
        if do_print:
            print(f'threshold = {threshold}')
        clustering = (F_out > threshold).to(torch.int8)

        algorithm_complexes = []
        for cluser_id in range(clustering.shape[1]):
            indices = torch.where(clustering[:, cluser_id] ==1)[0]
            if len(indices) > 0:
                alg_complex = []
                for protein_idx in indices.tolist():
                    protein_name = ppi_data_loader.id_to_protein_name(protein_idx)
                    alg_complex.append(protein_name)
                algorithm_complexes.append(alg_complex)

        if do_print:
            print('Number of clusters', len(algorithm_complexes))
            print('Number of clusters with one protein', sum([len(c) <= 1 for c in algorithm_complexes]))
        algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
        if do_print:
            print('Number of algorithm complexes:', len(algorithm_complexes))
        try:
            result = evaluator.evalute(algorithm_complexes)
        except:
            result = {
                'Precision': -1,
                'Recall': -1,
                'Acc': -1,
                'F1': -1,
                'NCP':-1,
                'NCB': -1,
            }

        if do_print:
            print(result)
            print('#'*100)
        if result['F1'] > max_f1:
            max_f1 = result['F1']
            best_threshold = threshold
            best_result = result

    best_result['best_threshold'] = best_threshold
    return best_result

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
A[edge_index[0] , edge_index[1]] = torch.tensor(ppi_data_loader.weights, dtype=torch.float32)
# A[data.edge_index[0] , data.edge_index[1]] = 1

# Berpo Decoder initialization
decoder = BerpoDecoder(num_nodes, A.sum().item(), balance_loss=False)

# evaluator class
evaluator = Evaluation('datasets/golden standard/ada_ppi.txt', ppi_data_loader)
evaluator.filter_reference_complex(filtering_method='just_keep_dataset_proteins')

history = {
    'loss':[],
    'F1':[]
}

best_f1 = -1
best_result_save = None

# train
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    F_out = model(bp_inputs, bp_mask, mf_inputs, mf_mask, cc_inputs, cc_mask, edge_index)
    loss= decoder.loss_full_weighted(F_out, A)
    loss.backward()
    optimizer.step()
    best_result = evaluate_model(model, evaluator, bp_inputs, bp_mask, mf_inputs, mf_mask, cc_inputs, cc_mask, edge_index, ppi_data_loader, do_print=False)
    model.train()
    history['loss'].append(loss.item())
    history['F1'].append(best_result['F1'])
    print(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}, F1: {best_result['F1']:.4f}')

    if best_result['F1'] > best_f1:
        best_f1 = best_result['F1']
        best_result_save = best_result
        print(f'# Best F1 updated to {best_result_save["F1"]}')
        torch.save(model.state_dict(), os.path.join(base_dir, 'weights', f'{file_name}.pt'))

with open(os.path.join(base_dir, 'results', f'{file_name}.json'), 'w') as f:
    json.dump(best_result_save, f)
    
plt.figure()
plt.plot(history['loss'], label='Loss')
plt.plot(history['F1'], label='F1')
plt.legend()
plt.savefig(os.path.join(base_dir, 'plots',f'{file_name}.jpg'))
# %%
