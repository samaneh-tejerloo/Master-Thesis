#%%
import pandas as pd
import os
from tqdm import tqdm
#%%
# Converting AdaPPI‌ txt datasets to csv file with weights
# read tadw_sc dataset
TADW_SC_DS_PATH = 'datasets/tadw-sc/krogan-extended/krogan-extended.csv'
tadw_dataset = pd.read_csv(TADW_SC_DS_PATH, index_col=0)

ADA_PPI_DS_PATH = 'datasets/AdaPPI/Krogan14k/krogan14k.txt'

with open(ADA_PPI_DS_PATH, 'r') as f:
    ada_ppi_dataset = f.readlines()
#%%
data = {
    'protein1':[],
    'protein2': [],
    'weight':[]
}
for line in tqdm(ada_ppi_dataset):
    p1,p2 = line.strip().split('\t')
    weight = None
    row = tadw_dataset[(tadw_dataset['protein1'] == p1) & (tadw_dataset['protein2'] == p2)]
    if len(row) == 0:
        row = tadw_dataset[(tadw_dataset['protein1'] == p2) & (tadw_dataset['protein2'] == p1)]
        if len(row) != 0:
            weight = row['weight'].values.item()
    else:
        weight = row['weight'].values.item()

    if weight is None:
        print(f'Interaction between {p1} and {p2} is not in TADW_SC dataset.')
        weight = tadw_dataset['weight'].mean().item()
    data['protein1'].append(p1)
    data['protein2'].append(p2)
    data['weight'].append(weight)

ada_ppi_dataset_df = pd.DataFrame(data)

ada_ppi_dataset_df.to_csv(ADA_PPI_DS_PATH.replace('txt','csv'))
#%%
pd.read_csv(ADA_PPI_DS_PATH.replace('txt','csv'), index_col=0)
# %%
