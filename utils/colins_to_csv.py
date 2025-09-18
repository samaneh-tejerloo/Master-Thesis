# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:29:11 2024

@author: tejer
"""

import pandas as pd

protein1_list = []
protein2_list = []
weight_list = []

with open('krogan2006_extended.txt', 'r') as file:
    for line in file:
        items = line.split('\t')
        protein1 = items[0]
        protein2 = items[1]
        weight = float(items[2])
        protein1_list.append(protein1)
        protein2_list.append(protein2)
        weight_list.append(weight)        
        #print(f'{protein1} with {protein2} has interaction with weight = {weight}')
        #break
  
data = {'protein1':protein1_list, 'protein2':protein2_list,\
        'weight':weight_list}

df = pd.DataFrame(data)
print(df)

df.to_csv('krogan-extended.csv')
#%%
