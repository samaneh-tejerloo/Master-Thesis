import torch
from evaluate import Evaluation
from itertools import chain
from dataset import PPIDataLoadingUtil
from models import SimpleGNN
from train.utils import process_features
import torch_geometric.nn as gnn
from torch_geometric.data import Data

# %%
path = "datasets/tadw-sc/krogan-core/krogan-core.csv"
embedding_dataset = PPIDataLoadingUtil(path, load_embeddings=True, load_weights=True)
one_hot_dataset = PPIDataLoadingUtil(path, load_embeddings=False, load_weights=True)
# %%
edge_index = torch.tensor(one_hot_dataset.edges_index).long().T
one_hot_features = one_hot_dataset.get_features(
    "one_hot", name_spaces=["BP", "MF", "CC"]
)
one_hot_features = torch.tensor(one_hot_features, dtype=torch.float32)

# %%
bp_features = embedding_dataset.get_features(type="embedding", name_spaces=["BP"])
mf_features = embedding_dataset.get_features(type="embedding", name_spaces=["MF"])
cc_features = embedding_dataset.get_features(type="embedding", name_spaces=["CC"])

bp_features = process_features(bp_features, edge_index)
mf_features = process_features(mf_features, edge_index)
cc_features = process_features(cc_features, edge_index)
embedding_features = torch.concat([bp_features, mf_features, cc_features], dim=-1)
# %%
model_one_hot = SimpleGNN(one_hot_features.shape[1], 512, 512, 2, gnn.GATConv)
model_embeddings = SimpleGNN(embedding_features.shape[1], 512, 512, 2, gnn.SuperGATConv)
# %%
model_one_hot.load_state_dict(torch.load("logs/one_hot.pt"))
model_one_hot.eval()
# %%

model_embeddings.load_state_dict(
    torch.load(
        "logs/krogan_core_SimpleGNN_SuperGAT_2_layers_4_heads_relu_BP_MF_CC_embedding.pt"
    )
)
model_embeddings.eval()
# %%

data_one_hot = Data(one_hot_features, edge_index)
data_embedding = Data(embedding_features, edge_index)
with torch.no_grad():
    F_one_hot = model_one_hot(data_one_hot)
    F_embedding = model_embeddings(data_embedding)
# %%


def get_algorithm_complexes(F_out, threshold=0.3):
    clustering = (F_out > threshold).to(torch.int8)

    algorithm_complexes = []
    for cluser_id in range(clustering.shape[1]):
        indices = torch.where(clustering[:, cluser_id] == 1)[0]
        if len(indices) > 0:
            alg_complex = []
            for protein_idx in indices.tolist():
                protein_name = one_hot_dataset.id_to_protein_name(protein_idx)
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


# %%

one_hot_complexes = get_algorithm_complexes(F_one_hot, threshold=0.2)
embeddings_complexes = get_algorithm_complexes(F_embedding, threshold=0.3)
# %%
evaluator = Evaluation(ppi_data_loader=one_hot_dataset)
evaluator.filter_reference_complex(filtering_method="just_keep_dataset_proteins")
# %%
one_hot_complexes = merge_unique([one_hot_complexes])
embeddings_complexes = merge_unique([embeddings_complexes])
all_complexes = merge_unique([one_hot_complexes, embeddings_complexes])
# %%
len(one_hot_complexes)
len(embeddings_complexes)
len(all_complexes)
# %%
evaluator.evalute(one_hot_complexes)
evaluator.evalute(embeddings_complexes)
evaluator.evalute(all_complexes)
