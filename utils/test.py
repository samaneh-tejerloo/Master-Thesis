#%%
from dataset import PPIDataLoadingUtil

dataset = PPIDataLoadingUtil('datasets/AdaPPI/COLLINS/collins.csv', load_embeddings=True, ada_ppi_dataset=True)
#%%
tadw_dataset = PPIDataLoadingUtil('datasets/tadw-sc/collins/colins.csv',  load_embeddings=True, ada_ppi_dataset=False)
#%%
tadw_dataset.get_go_terms_and_embeddings(tadw_dataset.proteins[0], load_embeddings=True)
#%%
dataset.get_go_terms_and_embeddings(dataset.proteins[0], load_embeddings=True)
#%%
tadw_dataset.get_features(type='one_hot').sum(axis=0)
#%%
dataset.get_features(type='one_hot').shape
# %%
protein_semantic_name = dataset.proteins[0]
load_embeddings = False
csv_path = dataset.csv_path
# %%
with open(csv_path.replace('.csv','_go_information.txt'), 'r') as f:
    go_terms_data = f.readlines()

go_terms_dict = {}
for line in go_terms_data:
    parts = line.split()
    go_terms_dict[parts[0]] = parts[1:]
# %%
# checking the inconsitency in the paper for number of go terms reported

with open('datasets/AdaPPI/Krogan14k/krogan14k_go_information.txt', 'r') as f:
    go_terms_data = f.readlines()

go_terms_dict = {}
for line in go_terms_data:
    parts = line.split()
    go_terms_dict[parts[0]] = parts[1:]

len(set([go_term for protein in go_terms_dict for go_term in go_terms_dict[protein]]))
# %%
