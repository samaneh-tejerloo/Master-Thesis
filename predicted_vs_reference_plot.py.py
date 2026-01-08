#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 05:35:57 2026

@author: user
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dataset import PPIDataLoadingUtil
from evaluate import Evaluation
#%%
dataset = PPIDataLoadingUtil('datasets/tadw-sc/collins_2007/colins2007.csv')
evaluator = Evaluation(ppi_data_loader=dataset)
evaluator.filter_reference_complex(filtering_method="just_keep_dataset_proteins")
reference_complexes = evaluator.filtered_complexes
#%%
with open('logs/algorithm_complexes/collins2007.txt') as f:
    lines = f.read().split('\n')
#%%
algorithm_complexes = [line.split() for line in lines]
#%%
len_algorithm = np.array([len(c) for c in algorithm_complexes])
len_reference = np.array([len(c) for c in reference_complexes])
#%%
data = [len_reference, len_algorithm]

plt.figure(figsize=(4, 6))
sns.violinplot(
    data=data,
    inner="box",      # shows white box inside (like your image)
    cut=0,            # do not extend beyond data range
    linewidth=1
)

plt.xticks([0, 1], ["Reference", "Algorithm"])
plt.ylabel("Length Complexes")
plt.tight_layout()
plt.show()