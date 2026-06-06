import torch
from evaluate import Evaluation
from itertools import chain
from dataset import PPIDataLoadingUtil
from models import SimpleGNN
from train.utils import process_features
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import numpy as np
#%%
def print_results(result):
    for key, value in result.items():
        print(key, ":", np.round(value, 3))
def get_algorithm_complexes(F_out, dataset, threshold=0.3):
    clustering = (F_out > threshold).to(torch.int8)

    algorithm_complexes = []
    for cluser_id in range(clustering.shape[1]):
        indices = torch.where(clustering[:, cluser_id] == 1)[0]
        if len(indices) > 0:
            alg_complex = []
            for protein_idx in indices.tolist():
                protein_name = dataset.id_to_protein_name(protein_idx)
                alg_complex.append(protein_name)
            algorithm_complexes.append(alg_complex)

    print("Number of clusters", len(algorithm_complexes))
    print(
        "Number of clusters with one protein",
        sum([len(c) <= 1 for c in algorithm_complexes]),
    )
    algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
    print("Number of algorithm complexes:", len(algorithm_complexes))
    return algorithm_complexes


def merge_unique(lists):
    all_groups = chain(*lists)
    uniq = set(frozenset(group) for group in all_groups)
    return [sorted(list(g)) for g in uniq]
#%%
def calculate_metric_without_dup(dataset_path, model_weight, is_embedding_model):
    if is_embedding_model:
        dataset = PPIDataLoadingUtil(dataset_path, load_embeddings=True, load_weights=True)
    else:
        dataset = PPIDataLoadingUtil(dataset_path, load_embeddings=False, load_weights=True)
        
    edge_index = torch.tensor(dataset.edges_index).long().T
    
    if is_embedding_model:
        bp_features = dataset.get_features(type="embedding", name_spaces=["BP"])
        mf_features = dataset.get_features(type="embedding", name_spaces=["MF"])
        cc_features = dataset.get_features(type="embedding", name_spaces=["CC"])
    
        bp_features = process_features(bp_features, edge_index)
        mf_features = process_features(mf_features, edge_index)
        cc_features = process_features(cc_features, edge_index)
        features = torch.concat([bp_features, mf_features, cc_features], dim=-1)
    else:
        one_hot_features = dataset.get_features(
            "one_hot", name_spaces=["BP", "MF"]
        )
        features = torch.tensor(one_hot_features, dtype=torch.float32)

    model = SimpleGNN(features.shape[1], 512, 512, 2, gnn.GATConv)

    model.load_state_dict(torch.load(model_weight))
    model.eval()
    data = Data(features, edge_index)
    with torch.no_grad():
        F_out = model(data)

    evaluator = Evaluation(ppi_data_loader=dataset)
    evaluator.filter_reference_complex(filtering_method="just_keep_dataset_proteins")
    
    algorithm_complexes = get_algorithm_complexes(F_out, dataset, threshold=0.3)
    print(f'{len(algorithm_complexes)} found complexes by algorithm.')
    all_complexes = merge_unique([algorithm_complexes])
    print(f'{len(all_complexes)} complexes after removing duplicated ones.')
    all_complexes = [c for c in all_complexes if len(c) > 2]
    print(f'{len(all_complexes)} complexes after removing complexes with less than 3 complexes')

    result = evaluator.evalute(all_complexes)
    print_results(result)
#%%

dataset_path = "datasets/tadw-sc/DIP/DIP.csv"
model_weight = 'logs/embeddings_gelu/weights/metric_DIP_SimpleGNN_GAT_2-layers_4-heads_gelu_BP_MF_CC_512_5000_embedding_weighted_best_modularity.pt'
is_embedding_model = True
calculate_metric_without_dup(dataset_path, model_weight, is_embedding_model)
