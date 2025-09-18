# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:00:57 2025

@author: tejer
"""

from goatools.anno.gaf_reader import GafReader
import json
ogaf = GafReader("datasets/sgd.gaf")
#%%
ns2assc = ogaf.get_ns2assc()
#%%
with open('datasets/colins_semantic_name_to_sgd_id.json') as f:
    semantic_name_to_sgd_id = json.load(f)
#%%
with open('datasets/colins_sgd_id_to_semantic_name.json') as f:
    sgd_id_to_semantic_name = json.load(f)
#%%

def get_go_terms(protein_semantic_name):
    try:
        mf_list = list(ns2assc['MF'][semantic_name_to_sgd_id[protein_semantic_name]])
    except:
        mf_list = []
    try:
        bp_list = list(ns2assc['BP'][semantic_name_to_sgd_id[protein_semantic_name]])
    except:
        bp_list = []
    try:
        cc_list = list(ns2assc['CC'][semantic_name_to_sgd_id[protein_semantic_name]])
    except:
        cc_list = []
    go_terms = mf_list + bp_list + cc_list
    
    return go_terms

protein_semantic_name = 'YML041C'
get_go_terms(protein_semantic_name)
#%%