# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:30:19 2025

@author: Samaneh
"""

#%%
import torch
import torch.nn as nn
#%%
n = 200
k = 3
d = 256
x = torch.rand(k,n,d)
#%%
lstm = nn.LSTM(input_size=d, hidden_size=d, num_layers=1, batch_first=False, bidirectional=True)
#%%
out, (h_n, c_n) = lstm(x)
#%%
sum([p.numel() for p in lstm.parameters()])