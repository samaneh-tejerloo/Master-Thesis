#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
# %%
file_path = 'datasets/AdaPPI/DIP/dip.txt'
dataset_name = os.path.basename(file_path).split('.')[0]
dir_name = os.path.dirname(file_path)
with open(file_path, 'r') as f:
    lines = f.readlines()
# %%
protein1 = []
protein2 = []

for line in lines:
    p1, p2 = line.strip().split('\t')
    protein1.append(p1)
    protein2.append(p2)
# %%
all_proteins = sorted(list(set(protein1 + protein2)))
# %%
proteins_list = ','.join(all_proteins)
with open(os.path.join(dir_name, f'{dataset_name}_proteins_list.txt'), 'w') as f:
    f.write(proteins_list)
# %%
sgd_results = pd.read_csv(os.path.join(dir_name,'sgd_results.csv'))
#%%
semantic_name_to_sgd_id = {}

for p in all_proteins:
    try:
        sgd_id = sgd_results[sgd_results['Gene > Systematic Name'] == p]['Gene > Primary DBID'].item()
        semantic_name_to_sgd_id[p] = sgd_id[4:]
    except Exception as e:
        print(f'SGD id for protein {p} has not been found!')
#%%
sgd_id_to_semantic_name = {}

for key,value in semantic_name_to_sgd_id.items():
    sgd_id_to_semantic_name[value] = key
#%%
import json

with open(os.path.join(dir_name,'semantic_name_to_sgd_id.json'),'w') as f:
    json.dump(semantic_name_to_sgd_id, f)


with open(os.path.join(dir_name,'sgd_id_to_semantic_name.json'),'w') as f:
    json.dump(sgd_id_to_semantic_name, f)
#%%
ppi_df = pd.DataFrame({'protein1':protein1, 'protein2':protein2})
ppi_df.to_csv(os.path.join(dir_name,f'{dataset_name}.csv'))
#%%
from dataset import PPIDataLoadingUtil

dataset = PPIDataLoadingUtil(os.path.join(dir_name,f'{dataset_name}.csv'), load_embeddings=False, load_weights=False, ada_ppi_dataset=False)
# %%
features = dataset.get_features('one_hot',name_spaces=['BP','MF','CC'])
# %%
eps = 1e-4
features = features / (np.linalg.norm(features, axis=1).reshape(-1,1) + eps)
#%%
weights = []
for idx, row in tqdm(ppi_df.iterrows(), total=len(ppi_df)):
    p1 = row['protein1']
    p2 = row['protein2']
    p1_idx = dataset.protein_name_to_id(p1)
    p2_idx = dataset.protein_name_to_id(p2)

    weight = (features[p1_idx] @ features[p2_idx]).item() + eps

    weights.append(weight)
# %%
ppi_df['weight'] = weights
# %%
ppi_df.to_csv(os.path.join(dir_name,f'{dataset_name}.csv'))
# %%
