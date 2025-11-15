# %%
from itertools import chain
from dataset import PPIDataLoadingUtil
from torch_geometric.data import Data
from tqdm import tqdm
from evaluate import Evaluation
from constants import SGD_GOLD_STANDARD_PATH
import numpy as np
from models import SimpleGNN
import torch

# %%
# dataset = "/home/shervin/Royam/Master-Thesis/datasets/tadw-sc/biogrid/biogrid.csv"
# dataset = "/home/shervin/Royam/Master-Thesis/datasets/tadw-sc/collins_2007/colins2007.csv"
dataset = "datasets/tadw-sc/krogan-core/krogan-core.csv"
# dataset = "/home/shervin/Royam/Master-Thesis/datasets/tadw-sc/krogan-extended/krogan-extended.csv"
# dataset = "/home/shervin/Royam/Master-Thesis/datasets/tadw-sc/DIP/DIP.csv"
weights = (
    "logs/weights/krogan-core_SimpleGNN_GAT_2-layers_4-heads_relu_BP_MF_512_2000.pt"
)
ppi_data_loader = PPIDataLoadingUtil(
    dataset, load_embeddings=False, load_weights=True, ada_ppi_dataset=False
)

edge_index = torch.LongTensor(ppi_data_loader.edges_index).T
features = ppi_data_loader.get_features(type="one_hot", name_spaces=["BP", "MF"])
embedding_dim = features.shape[1]
features = torch.tensor(features, dtype=torch.float32)

model = SimpleGNN(
    embedding_dim=embedding_dim, intermediate_dim=512, encoding_dim=512, heads=4
)

model.load_state_dict(torch.load(weights))

model.eval()

data = Data(x=features, edge_index=edge_index)

with torch.no_grad():
    F_out = model(data)


def get_algorithm_complexes(F_out, threshold=0.3):
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


# algorithm_complexes_1 = get_algorithm_complexes(F_out, threshold=0.1)
algorithm_complexes_2 = get_algorithm_complexes(F_out, threshold=0.2)
algorithm_complexes_3 = get_algorithm_complexes(F_out, threshold=0.3)
algorithm_complexes_4 = get_algorithm_complexes(F_out, threshold=0.4)
algorithm_complexes_5 = get_algorithm_complexes(F_out, threshold=0.5)
algorithm_complexes_6 = get_algorithm_complexes(F_out, threshold=0.6)
algorithm_complexes_7 = get_algorithm_complexes(F_out, threshold=0.7)

all_complexes = [
    algorithm_complexes_2,
    algorithm_complexes_3,
    algorithm_complexes_4,
    algorithm_complexes_5,
    algorithm_complexes_6,
    algorithm_complexes_7,
]
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# %%
for i in range(len(all_complexes)):
    complexes = all_complexes[: i + 1]
    threshes = thresholds[: i + 1]
    complexes = merge_unique(complexes)
    print(threshes)
    print(f"Number of unique complexes: {len(complexes)}")
    evaluator = Evaluation(SGD_GOLD_STANDARD_PATH, ppi_data_loader)
    evaluator.filter_reference_complex(filtering_method="just_keep_dataset_proteins")
    result = evaluator.evalute(complexes)
    print(result)

print("Testing the 0.2 and 0.7 thresholds")
complexes = [all_complexes[0], all_complexes[-1]]
complexes = merge_unique(complexes)
print(f"Number of unique complexes: {len(complexes)}")
evaluator = Evaluation(SGD_GOLD_STANDARD_PATH, ppi_data_loader)
evaluator.filter_reference_complex(filtering_method="just_keep_dataset_proteins")
result = evaluator.evalute(complexes)
print(result)
# %%
