# %%
import sys
sys.path.append('/home/user/Master-Thesis')
from dataset import PPIDataLoadingUtil
from torch_geometric.data import Data
from nocd_decoder import BerpoDecoder
from tqdm import tqdm
from evaluate import Evaluation
from constants import SGD_GOLD_STANDARD_PATH
import numpy as np
from models import SimpleGNN
import pandas as pd
import torch
import torch_geometric.nn as gnn
import torch.nn.functional as F
import json
import os
from train.utils import process_features
import matplotlib.pyplot as plt
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

experiment_name = 'embeddings_gelu'
base_dir = os.path.join("logs", experiment_name)

dataset = "datasets/tadw-sc/collins_2007/colins2007.csv"


def convert_clusters_name_to_clusters_id(clusters, dataset):
    clusters_id = []
    for cluster in clusters:
        cluster_id = []
        for p in cluster:
            cluster_id.append(dataset.protein_name_to_id(p))
        clusters_id.append(cluster_id)
    return clusters_id


def calculate_modularity(F_out, threshold, data, ppi_data_loader):
    clustering = F_out > threshold
    clustering = clustering.float()
    n = data.x.shape[0]
    m = data.edge_index.shape[1] // 2
    A = torch.zeros(n, n).to(device)
    A[data.edge_index[0], data.edge_index[1]] = torch.tensor(ppi_data_loader.weights).to(device)
    degree_vector = A.sum(axis=0).reshape(-1, 1)
    K = (degree_vector @ degree_vector.T) / (2 * m)

    o_vector = clustering.sum(dim=1, keepdim=True)
    O = o_vector @ o_vector.T

    delta = (clustering @ clustering.T) > 0
    O_safe = O.clone()
    O_safe[O_safe == 0] = 1
    modularity = 1 / (2 * m) * ((1 / O_safe) * (A - K) * delta).sum()
    return float(modularity)

def evaluate_model(model, evaluator, data, ppi_data_loader, do_print=False):
    # evaluating the model
    n = data.x.shape[0]
    A = torch.zeros(n, n).to(device)
    A[data.edge_index[0], data.edge_index[1]] = 1

    model.eval()
    with torch.no_grad():
        F_out = model(data)

    result = {
        "Precision": -1,
        "Recall": -1,
        "Acc": -1,
        "F1": -1,
        "NCP": -1,
        "NCB": -1,
        "modularity": -1,
    }
    
    threshold = 0.3
    if do_print:
        print(f"threshold = {threshold}")
    clustering = (F_out > threshold).to(torch.int8)

    algorithm_complexes = []
    for cluser_id in range(clustering.shape[1]):
        indices = torch.where(clustering[:, cluser_id] == 1)[0]
        if len(indices) > 0:
            alg_complex = []
            for protein_idx in indices.tolist():
                protein_name = ppi_data_loader.id_to_protein_name(protein_idx)
                alg_complex.append(protein_name)
            algorithm_complexes.append(alg_complex)

    if do_print:
        print("Number of clusters", len(algorithm_complexes))
        print(
            "Number of clusters with one protein",
            sum([len(c) <= 1 for c in algorithm_complexes]),
        )
    algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
    clusters_id = convert_clusters_name_to_clusters_id(
        algorithm_complexes, ppi_data_loader
    )

    if do_print:
        print("Number of algorithm complexes:", len(algorithm_complexes))
    try:
        result = evaluator.evalute(algorithm_complexes)
    except:
        pass
    
    
    try:
        modularity = calculate_modularity(F_out, threshold, data, ppi_data_loader)
        result["modularity"] = float(modularity)
    except Exception as e:
        print("modularity error:", e)
    
    if do_print:
        print(result)
        print("#" * 100)

    return result


def train_config(
    model,
    layers,
    layer_type,
    heads,
    feature_type,
    name_space,
    activation_function,
    dataset,
    intermediate_dim=512,
    epochs=2000,
    weighted=True,
    lambda_a2=0,
    test_mode=False,
):
    print("#" * 10, "Config", "#" * 10)
    print(f"dataset:\t {dataset}")
    print(f"model:\t {model}")
    print(f"layers:\t {layers}")
    print(f"layer_type:\t {layer_type}")
    print(f"heads:\t {heads}")
    print(f"feature_type:\t {feature_type}")
    print(f"name_space:\t {name_space}")
    print(f"activation_function:\t {activation_function}")
    print(f"intermediate_dim:\t {intermediate_dim}")
    print(f"epochs:\t {epochs}")
    
    if weighted:
        file_name = f"metric_{os.path.basename(dataset).split('.')[0]}_{model}_{layer_type}_{layers}-layers_{heads}-heads_{activation_function}_{'_'.join(name_space)}_{intermediate_dim}_{epochs}_{feature_type}_weighted"
    else:
        file_name = f"metric_{os.path.basename(dataset).split('.')[0]}_{model}_{layer_type}_{layers}-layers_{heads}-heads_{activation_function}_{'_'.join(name_space)}_{intermediate_dim}_{epochs}_{feature_type}_lambda_{lambda_a2}"
    load_embeddings = False
    if feature_type == "embedding":
        load_embeddings = True

    ppi_data_loader = PPIDataLoadingUtil(
        dataset,
        load_embeddings=load_embeddings,
        load_weights=True,
        ada_ppi_dataset=False,
    )
    edge_index = torch.LongTensor(ppi_data_loader.edges_index).T.to(device)

    if feature_type == "one_hot":
        features = ppi_data_loader.get_features(
            type=feature_type, name_spaces=name_space
        )
    elif feature_type == "embedding":
        features_list = []
        for ns in name_space:
            features = ppi_data_loader.get_features(type=feature_type, name_spaces=[ns])
            features = process_features(features, edge_index)
            features_list.append(features)
        features = torch.concat(features_list, dim=-1)

    features = torch.tensor(features, dtype=torch.float32).to(device)
    print(f"features_shape: {features.shape}")
    data = Data(x=features, edge_index=edge_index)

    embedding_dim = data.num_features

    # mapping activation function
    if activation_function == 'relu':
        activation_function = F.relu
    elif activation_function == 'gelu':
        activation_function = F.gelu
    elif activation_function == 'elu':
        activation_function = F.elu
    
    # initializing the model
    if model == "SimpleGNN":
        if layer_type == "GCN" and heads is None:
            model = SimpleGNN(
                embedding_dim=embedding_dim,
                intermediate_dim=intermediate_dim,
                encoding_dim=intermediate_dim,
                n_layers=layers,
                layer_module=gnn.GCNConv,
                activation=activation_function,
            )
        elif layer_type == "GAT" and heads is not None:
            model = SimpleGNN(
                embedding_dim=embedding_dim,
                intermediate_dim=intermediate_dim,
                encoding_dim=intermediate_dim,
                n_layers=layers,
                layer_module=gnn.GATConv,
                activation=activation_function,
                heads=heads,
            )
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float32).to(device)
    if weighted:
        A[data.edge_index[0], data.edge_index[1]] = torch.tensor(
            ppi_data_loader.weights, dtype=torch.float32
            ).to(device)
        # Berpo Decoder initialization
        decoder = BerpoDecoder(data.num_nodes, A.sum().item(), balance_loss=False)
    else:
        A[data.edge_index[0], data.edge_index[1]] = 1
        A_2 = A @ A
        A_2[A_2 > 0] = 1
        for i in range(A.shape[0]):
            A_2[i,i] = 0
        decoder = BerpoDecoder(data.num_nodes, A.sum().item(), balance_loss=False)
        decoder_2 = BerpoDecoder(data.num_nodes, A_2.sum().item(), balance_loss=False)
        
    # A[data.edge_index[0] , data.edge_index[1]] = 1


    # evaluator class
    evaluator = Evaluation("datasets/golden standard/ada_ppi.txt", ppi_data_loader)
    evaluator.filter_reference_complex(filtering_method="just_keep_dataset_proteins")

    history = {
        "loss": [],
        "F1": [],
        "modularity": [],
    }

    best_f1 = -1
    best_result_save = None
    best_modularity = -1

    # train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        F_out = model(data)
        if weighted:
            loss = decoder.loss_full_weighted(F_out, A)
        else:
            if lambda_a2 == 0:
                loss_1 = decoder.loss_full(F_out, A)
                loss = loss_1
            elif lambda_a2 == 1:
                loss_2 = decoder_2.loss_full(F_out, A_2)
                loss = loss_2
            else:
                loss_1 = decoder.loss_full(F_out, A)
                loss_2 = decoder_2.loss_full(F_out, A_2)
                loss = (1-lambda_a2) * loss_1 + (lambda_a2) * loss_2
        
        loss.backward()
        optimizer.step()
        result = evaluate_model(model, evaluator, data, ppi_data_loader, do_print=False)
        model.train()
        history["loss"].append(loss.item())
        history["F1"].append(result["F1"])
        modularity = result["modularity"]
        history["modularity"].append(modularity)

        print(
            f"Epoch: {epoch + 1:02}/{epochs}, loss:{loss.item():.4f}, F1: {result['F1']:.4f}, modularity: {modularity:.4f}"
        )

        if result["modularity"] > best_modularity:
            best_modularity = result["modularity"]
            best_result_save = result.copy()

            best_result_save["loss"] = loss.item()
            best_result_save["epoch"] = epoch

            print(f"# Best modularity updated to {best_result_save['modularity']}")
            torch.save(
                model.state_dict(),
                os.path.join(base_dir, "weights", f"{file_name}_best_modularity.pt"),
            )

            if test_mode:
                break

        torch.save(
            model.state_dict(),
            os.path.join(base_dir, "weights", f"{file_name}_last.pt"),
        )

    with open(os.path.join(base_dir, "results", f"{file_name}_best.json"), "w") as f:
        json.dump(best_result_save, f)

    with open(os.path.join(base_dir, "history", f"{file_name}.json"), "w") as f:
        json.dump(history, f)

    # plt.figure()
    # plt.plot(history["loss"], label="Loss")
    # plt.plot(history["F1"], label="F1")
    # plt.plot(history["modularity"], label="modularity")
    # plt.legend()
    # plt.savefig(os.path.join(base_dir, "plots", f"{file_name}.jpg"))
    
    return best_result_save, history
# %%
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "weights"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "history"), exist_ok=True)
#%%
datasets = [
    'datasets/tadw-sc/collins_2007/colins2007.csv',
    'datasets/tadw-sc/krogan-core/krogan-core.csv',
    'datasets/tadw-sc/DIP/DIP.csv',
    'datasets/tadw-sc/krogan-extended/krogan-extended.csv',
    'datasets/tadw-sc/biogrid/biogrid.csv'
    ]
#activation_functions = ['relu', 'gelu', 'elu'] 
activation_functions = ['gelu'] 

for dataset in datasets:
    for activation_function in activation_functions:
        if 'DIP' in dataset or 'biogrid' in dataset:
            epochs = 5000
        else:
            epochs = 2000
        
        layer_type = 'GAT'
        num_heads = 4
       
        
        weighted = True
        lambda_a2 = 0
        
        
        best_result, history = train_config(
            "SimpleGNN",
            2,
            layer_type,
            num_heads,
            "embedding",
            ["BP", "MF", "CC"],
            activation_function,
            dataset,
            weighted=weighted,
            lambda_a2=lambda_a2,
            test_mode=False,
            epochs=epochs,
        )
        gc.collect()
#%%
