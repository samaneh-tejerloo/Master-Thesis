#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 04:44:57 2026

@author: user
"""

#%%
from dataset import PPIDataLoadingUtil
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
datasets = [
    'datasets/tadw-sc/collins_2007/colins2007.csv',
    'datasets/tadw-sc/krogan-core/krogan-core.csv',
    'datasets/tadw-sc/krogan-extended/krogan-extended.csv',
    'datasets/tadw-sc/DIP/DIP.csv',
    'datasets/tadw-sc/biogrid/biogrid.csv',
    ]
#%%
dataset_names = []
data = []
for dataset_path in datasets:
    dataset_name = os.path.basename(dataset_path).split('.')[0]
    dataset_binary = PPIDataLoadingUtil(dataset_path, load_embeddings=False, load_weights=True)
    binary_features = dataset_binary.get_features('one_hot', name_spaces=['MF','BP'])
    n = len(dataset_binary.proteins)
    A = np.zeros((n,n)).astype(np.int8)
    edge_indices = np.array(dataset_binary.edges_index)
    A[edge_indices[:,0], edge_indices[:,1]] = 1
    degrees = A.sum(axis=0)
    
    print("#"*20)
    print("#", dataset_name)
    print("#"*20)
    print('Protein nodes:', n)
    print('Interaction edges', edge_indices.shape[0] // 2)
    print('GO attibutes binary', binary_features.shape[1])
    print('GO attibutes embedding', 384)
    print('Average degree', np.round(degrees.mean(),3))
    print('Max degree', degrees.max())
    dataset_names.append(dataset_name)
    data.append(degrees)
#%%
