# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:39:27 2025

@author: tejer
"""
#%%
import pandas as pd

df = pd.read_csv('datasets/tadw-sc/collins_2007/colins2007.csv',index_col=0)
#%%
protein_1 = list(df['protein1'])
protein_2 = list(df['protein2'])
proteins = protein_1 + protein_2
#%%
proteins = set(proteins)
#%%
proteins = list(proteins)
#%%
string_proteins = ''

# classic method
'''for p in proteins:
    string_proteins = p + ',' + string_proteins'''

# pythonic method
string_proteins = ','.join(proteins)
print(string_proteins)
#%%
f = open('datasets/tadw-sc/collins_2007/colins_proteins_list.txt','w')
f.write(string_proteins)
f.close()
#%%
with open('datasets/colins_proteins_list.txt','w') as f:
    f.write(string_proteins)
#%%
sgd_results = pd.read_csv('datasets/tadw-sc/collins_2007/sgd_results.csv')
#%%
semantic_name_to_sgd_id = {}

for p in proteins:
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

with open('datasets/tadw-sc/collins_2007/semantic_name_to_sgd_id.json','w') as f:
    json.dump(semantic_name_to_sgd_id, f)


with open('datasets/tadw-sc/collins_2007/sgd_id_to_semantic_name.json','w') as f:
    json.dump(sgd_id_to_semantic_name, f)











# %%
