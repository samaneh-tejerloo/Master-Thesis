#%%
from dataset import PPIDataLoadingUtil
from torch_geometric.data import Data
from nocd_decoder import BerpoDecoder
from tqdm import tqdm
from evaluate import Evaluation
from constants import SGD_GOLD_STANDARD_PATH
import numpy as np
from models import SimpleGNN, JKNetConcat, JKNetLSTMAttention, JKNetMaxPooling, JKNetMultiHeadAttention
import pandas as pd
import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F
import json
import os
from train.utils import process_features
base_dir = 'logs'
#%%
models=  ['SimpleGNN','JKNetConcat', 'JKNetMaxPooling', 'JKNetLSTMAttention-bidirectional','JKNetLSTMAttention-unidirectional']
layers = [2, 4, 8]
layer_types = ['GCN','GAT']
heads = [2,4,8]
feature_type = 'one_hot'
name_spaces = [['BP'], ['BP','MF'], ['BP','MF','CC']]
activation_function = ['relu','leaky_relu', 'gelu', 'elu']
dataset = 'datasets/tadw-sc/krogan-core/krogan-core.csv'
#%%
def train_config(model, layers, layer_type, heads, feature_type, name_space, activation_function, dataset, intermediate_dim=512, epochs=2000):

    print('#'*10,'Config','#'*10)
    print(f'dataset:\t {dataset}')
    print(f'model:\t {model}')
    print(f'layers:\t {layers}')
    print(f'layer_type:\t {layer_type}')
    print(f'heads:\t {heads}')
    print(f'feature_type:\t {feature_type}')
    print(f'name_space:\t {name_space}')
    print(f'activation_function:\t {activation_function}')
    print(f'intermediate_dim:\t {intermediate_dim}')
    print(f'epochs:\t {epochs}')

    file_name = f'{model}_{layer_type}_{layers}layers_{heads}heads_{activation_function}_{'_'.join(name_space)}_{intermediate_dim}_{epochs}'

    ppi_data_loader = PPIDataLoadingUtil(dataset, load_embeddings=False, load_weights=True, ada_ppi_dataset=False)
    edge_index = torch.LongTensor(ppi_data_loader.edges_index).T

    if feature_type == 'one_hot':
        features = ppi_data_loader.get_features(type=feature_type, name_spaces=name_space)
    elif feature_type == 'embedding':
        features_list = []
        for ns in name_space:
            features = ppi_data_loader.get_features(type=feature_type, name_spaces=[ns])
            features = process_features(features, edge_index)
            features_list.append(features)
        features = torch.concat(features_list, dim=-1)
    

    features = torch.tensor(features, dtype=torch.float32)
    print(features.shape)
    data = Data(x=features, edge_index=edge_index)

    embedding_dim = data.num_features

    # mapping activation function
    if activation_function == 'relu':
        activation_function = F.relu
    elif activation_function == 'leaky_relu':
        activation_function = F.leaky_relu
    elif activation_function == 'gelu':
        activation_function = F.gelu
    elif activation_function == 'elu':
        activation_function = F.elu

    # initializing the model
    if model == 'SimpleGNN':
        if layer_type == 'GCN' and heads is None:
            model = SimpleGNN(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GCNConv, activation=activation_function)
        elif layer_type == 'GAT' and heads is not None:
            model = SimpleGNN(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GATConv, activation=activation_function, heads=heads) 

        elif layer_type == 'GAT2' and heads is not None:
            model = SimpleGNN(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GATv2Conv, activation=activation_function, heads=heads) 
        else:
            print('Wrong model type...')
            return
    elif model == 'JKNetConcat':
        if layer_type == 'GCN' and heads is None:
            model = JKNetConcat(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GCNConv, activation=activation_function)
        elif layer_type == 'GAT' and heads is not None:
            model = JKNetConcat(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GATConv, activation=activation_function, heads=heads) 
        else:
            print('Wrong model type...')
            return
    elif model == 'JKNetMaxPooling':
        if layer_type == 'GCN' and heads is None:
            model = JKNetMaxPooling(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GCNConv, activation=activation_function)
        elif layer_type == 'GAT' and heads is not None:
            model = JKNetMaxPooling(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GATConv, activation=activation_function, heads=heads) 
        else:
            print('Wrong model type...')
            return
    elif model == 'JKNetLSTMAttention-bidirectional' or model=='JKNetLSTMAttention-unidirectional':
        bidirectional = True if 'bidirectional' in model else False

        if layer_type == 'GCN' and heads is None:
            model = JKNetLSTMAttention(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GCNConv, activation=activation_function, bidirectional=bidirectional)
        elif layer_type == 'GAT' and heads is not None:
            model = JKNetLSTMAttention(embedding_dim=embedding_dim, intermediate_dim=intermediate_dim, encoding_dim=intermediate_dim, n_layers=layers, layer_module=gnn.GATConv, activation=activation_function, heads=heads, bidirectional=bidirectional)
        else:
            print('Wrong model type...')
            return
    elif model == 'GIN':
        model = gnn.GIN(in_channels=embedding_dim, hidden_channels=intermediate_dim, num_layers=layers, out_channels=intermediate_dim, jk=None, act=activation_function)
    else:
        print('Wrong model type...')
        return
    

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float32)
    A[data.edge_index[0] , data.edge_index[1]] = torch.tensor(ppi_data_loader.weights, dtype=torch.float32)

    # Berpo Decoder initialization
    decoder = BerpoDecoder(data.num_nodes, A.sum().item(), balance_loss=False)


    # train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, gnn.GIN):
            F_out = model(data.x, data.edge_index)
            F_out = F.relu(F_out)
        else:
            F_out = model(data)
        loss= decoder.loss_full_weighted(F_out, A)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
    
    # evaluating the model
    model.eval()
    with torch.no_grad():
        if isinstance(model, gnn.GIN):
            F_out = model(data.x, data.edge_index)
            F_out = F.relu(F_out)
        else:
            F_out = model(data)
    evaluator = Evaluation('datasets/golden standard/ada_ppi.txt', ppi_data_loader)
    evaluator.filter_reference_complex(filtering_method='just_keep_dataset_proteins')

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

        print('Number of clusters', len(algorithm_complexes))
        print('Number of clusters with one protein', sum([len(c) <= 1 for c in algorithm_complexes]))
        algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
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

        print(result)
        print('#'*100)
        if result['F1'] > max_f1:
            max_f1 = result['F1']
            best_threshold = threshold
            best_result = result

    best_result['best_threshold'] = best_threshold
    with open(os.path.join(base_dir, 'results', f'{file_name}.json'), 'w') as f:
        json.dump(best_result, f)
    
    torch.save(model.state_dict(), os.path.join(base_dir, 'weights', f'{file_name}.pt'))

    return best_threshold, best_result
#%%
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, 'results'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'weights'), exist_ok=True)
#%%
best_threshold, best_result = train_config('SimpleGNN', 2, 'GAT2', 4, 'one_hot', ['BP','MF'], 'relu', dataset)
print('#'*10, f'Train finished best results best_threshold={best_threshold}', '#'*10)
print(best_result)
#%%
best_threshold, best_result = train_config(models[0], layers[0], layer_types[1], heads[1], feature_type, name_spaces[1], activation_function[0], dataset, epochs=100)
print('#'*10, f'Train finished best results best_threshold={best_threshold}', '#'*10)
print(best_result)
# %%
for name_space in name_spaces:
    for model in models:
        for activation_fn in activation_function:
            for layer in layers:
                for layer_type in layer_types:
                    if layer_type == 'GAT':
                        for head in heads:
                            best_threshold, best_result = train_config(model, layer, layer_type, head, feature_type, name_space, activation_fn, dataset, epochs=1)
                    else:
                        best_threshold, best_result = train_config(model, layer, layer_type, None, feature_type, name_space, activation_fn, dataset, epochs=1)
# %%
