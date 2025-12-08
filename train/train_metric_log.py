# %%
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

base_dir = "logs"

dataset = "datasets/tadw-sc/krogan-core/krogan-core.csv"


def calculate_modularity(F_out, threshold, data):
    clustering = F_out > threshold
    clustering = clustering.int().numpy()
    n = data.x.shape[0]
    m = data.edge_index.shape[1] // 2
    A = np.zeros((n, n))
    A[data.edge_index[0], data.edge_index[1]] = 1
    degree_vector = A.sum(axis=0).reshape(-1, 1)
    K = (degree_vector @ degree_vector.T) / (2 * m)

    o_vector = clustering.sum(axis=1, keepdims=True)
    O = o_vector @ o_vector.T

    delta = (clustering @ clustering.T) > 0
    O_safe = O.copy()
    O_safe[O_safe == 0] = 1
    modularity = 1 / (2 * m) * ((1 / O_safe) * (A - K) * delta).sum()
    print("here")
    return modularity


def evaluate_model(model, evaluator, data, ppi_data_loader, do_print=False):
    # evaluating the model
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
    if do_print:
        print("Number of algorithm complexes:", len(algorithm_complexes))
    try:
        result = evaluator.evalute(algorithm_complexes)
    except:
        pass

    try:
        modularity = calculate_modularity(F_out, threshold, data)
    except Exception as e:
        print(e)

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

    file_name = f"{os.path.basename(dataset).split('.')[0]}_{model}_{layer_type}_{layers}-layers_{heads}-heads_{activation_function}_{'_'.join(name_space)}_{intermediate_dim}_{epochs}"

    load_embeddings = False
    if feature_type == "embedding":
        load_embeddings = True

    ppi_data_loader = PPIDataLoadingUtil(
        dataset,
        load_embeddings=load_embeddings,
        load_weights=True,
        ada_ppi_dataset=False,
    )
    edge_index = torch.LongTensor(ppi_data_loader.edges_index).T

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

    features = torch.tensor(features, dtype=torch.float32)
    print(features.shape)
    data = Data(x=features, edge_index=edge_index)

    embedding_dim = data.num_features

    # mapping activation function
    activation_function = F.relu
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

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float32)
    A[data.edge_index[0], data.edge_index[1]] = torch.tensor(
        ppi_data_loader.weights, dtype=torch.float32
    )
    # A[data.edge_index[0] , data.edge_index[1]] = 1

    # Berpo Decoder initialization
    decoder = BerpoDecoder(data.num_nodes, A.sum().item(), balance_loss=False)

    # evaluator class
    evaluator = Evaluation("datasets/golden standard/ada_ppi.txt", ppi_data_loader)
    evaluator.filter_reference_complex(filtering_method="just_keep_dataset_proteins")

    history = {"loss": [], "F1": [], "diff": []}

    best_f1 = -1
    best_result_save = None
    prev_f_out = None

    # train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        F_out = model(data)
        loss = decoder.loss_full_weighted(F_out, A)
        loss.backward()
        optimizer.step()
        result = evaluate_model(model, evaluator, data, ppi_data_loader, do_print=False)
        model.train()
        history["loss"].append(loss.item())
        history["F1"].append(result["F1"])
        if prev_f_out is None:
            prev_f_out = F_out
            diff = -1
        else:
            diff = torch.abs(F_out - prev_f_out).sum()
            prev_f_out = F_out

        history["diff"].append(diff)
        modularity = result["modularity"]

        print(
            f"Epoch: {epoch + 1:02}/{epochs}, loss:{loss.item():.4f}, F1: {result['F1']:.4f}, diff: {diff:.4f}, modularity: {modularity:.4f}"
        )

        if result["F1"] > best_f1:
            best_f1 = result["F1"]
            best_result_save = result
            best_result_save["diff"] = diff
            best_result_save["loss"] = loss
            best_result_save["epoch"] = epoch
            print(f"# Best F1 updated to {best_result_save['F1']}")
            torch.save(
                model.state_dict(),
                os.path.join(base_dir, "weights", f"{file_name}_best.pt"),
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

    plt.figure()
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["F1"], label="F1")
    plt.legend()
    plt.savefig(os.path.join(base_dir, "plots", f"{file_name}.jpg"))

    return best_result_save, history


# %%
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "weights"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "history"), exist_ok=True)
# %%
best_result, history = train_config(
    "SimpleGNN",
    2,
    "GAT",
    4,
    "one_hot",
    ["BP", "MF"],
    "relu",
    dataset,
    test_mode=True,
)

print(
    "#" * 10,
    f"Train finished best results best_threshold={best_result['best_threshold']}",
    "#" * 10,
)
print(best_result)
