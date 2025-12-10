# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 11:27:38 2025

@author: Samaneh
"""

import json

with open('logs/history/krogan-extended_SimpleGNN_GAT_2-layers_4-heads_relu_BP_MF_512_2000.json') as f:
    history = json.load(f)
#%%
history.keys()
#%%
history['F1'][-1]
max(history['F1'])
#%%
history['diff'][-1]
min(history['diff'])
#%%
import numpy as np
#%%
a = np.array(history['diff'])
#%%
a[a == -1] = 1000
#%%
a.min()
#%%
a.argmin()
#%%
history['F1'][1633]
#%%
import matplotlib.pyplot as plt

plt.plot(history['F1'])
#%%
plt.plot(history['diff'])
#%%
np.argmax(history['F1'])
#%%
plt.plot(history['loss'])
#%%
history['loss'][929]
#%%
history['loss'][-1]
#%%
history['loss'][1310]
#%%
history['loss'][1838] - history['loss'][1839]
#%%
history['loss'][1837] - history['loss'][1838]
#%%
prev_loss = None
for i in range(2000):
    loss = history['loss'][i]
    if prev_loss is None:
        prev_loss = loss
    else:
        delta = prev_loss - loss
        prev_loss = loss
        if delta <= 1e-10:
            break
print(i)
print(history['F1'][i])
#%%
history['loss'][1838]