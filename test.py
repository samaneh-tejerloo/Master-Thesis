#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 05:57:05 2026

@author: user
"""

#%%
import numpy as np
import json

with open("logs/history/metric_colins2007_SimpleGNN_GAT_2-layers_4-heads_relu_BP_MF_CC_512_2000_embedding.json") as f:
    history = json.load(f)
#%%
np.argmax(history['F1'])

np.argmax(history['modularity'])
print(history['F1'][595])

np.argmax(history['density'])
history['F1'][154]

history['F1'][-1]
#%%
history['modularity'][-1]
#%%
max(history['modularity'])