#%%
from dataset import PPIDataLoadingUtil
from models import SimpleGCN
from torch_geometric.data import Data
import torch
import nocd
from tqdm import tqdm
from evaluate import Evaluation
from constants import SGD_GOLD_STANDARD_PATH
# %%
ppi_data_loader = PPIDataLoadingUtil('datasets/colins.csv')
# %%
features = ppi_data_loader.get_features(type='one_hot', name_spaces=['BP'])
features = torch.tensor(features, dtype=torch.float32)
edge_index = torch.LongTensor(ppi_data_loader.edges_index).T
#%%
data = Data(x=features, edge_index=edge_index)
# %%
model = SimpleGCN(data.num_features, 256, 40)
decoder = nocd.nn.BerpoDecoder(data.num_nodes, data.num_edges, balance_loss=True)
#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 

A = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.int8)
A[data.edge_index[0] , data.edge_index[1]] = 1

epochs = 100
model.train()
progress_bar = tqdm(range(epochs))
for epoch in progress_bar:
    optimizer.zero_grad()
    F_out = model(data)
    loss = decoder.loss_full(F_out, A.numpy())
    loss.backward()
    optimizer.step()
    progress_bar.set_description(f'Epoch: {epoch+1:02}/{epochs}, loss:{loss.item():.4f}')
#%%
model.eval()
with torch.no_grad():
    F_out = model(data)
threshold = 0.3
clustering = (F_out > threshold).to(torch.int8)
clustering.sum(dim=0)
# %%
algorithm_complexes = []
for cluser_id in range(clustering.shape[1]):
    indices = torch.where(clustering[:, cluser_id] ==1)[0]
    if len(indices) > 0:
        alg_complex = []
        for protein_idx in indices.tolist():
            protein_name = ppi_data_loader.id_to_protein_name(protein_idx)
            alg_complex.append(protein_name)
        algorithm_complexes.append(alg_complex)
# %%
evaluator = Evaluation(SGD_GOLD_STANDARD_PATH, ppi_data_loader)
# evaluator.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
evaluator.filter_reference_complex(filtering_method='all_proteins_in_dataset')
result = evaluator.evalute(algorithm_complexes, threshold_na=0.2)
result
# %%
