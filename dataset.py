#%%
import pandas as pd
from goatools.anno.gaf_reader import GafReader
from gensim.models import KeyedVectors
import numpy as np
import json
import torch
from torch_geometric.data import Data
import networkx as nx
import torch_geometric
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
import matplotlib.pyplot as plt
import nocd
from constants import BP_EMBEDDINGS_PATH, MF_EMBEDDINGS_PATH, CC_EMBEDDINGS_PATH, SGD_GAF_PATH 
import os
from tqdm import tqdm

class PPIDataLoadingUtil:
    def __init__(self, csv_path, load_embeddings=True, load_weights=True):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path, index_col=0)

        self.load_embeddings = load_embeddings
        if load_embeddings:
            self.bp_embeddings = KeyedVectors.load_word2vec_format(BP_EMBEDDINGS_PATH)
            self.cc_embeddings = KeyedVectors.load_word2vec_format(CC_EMBEDDINGS_PATH)
            self.mf_embeddings = KeyedVectors.load_word2vec_format(MF_EMBEDDINGS_PATH)
        
        ogaf = GafReader(SGD_GAF_PATH)
        SEMANTIC_NAME_TO_SGD_ID_JSON = os.path.join(os.path.dirname(csv_path),'semantic_name_to_sgd_id.json')
        with open(SEMANTIC_NAME_TO_SGD_ID_JSON) as f:
            self.semantic_name_to_sgd_id = json.load(f)
        
        self.ns2assc = ogaf.get_ns2assc()
        self.proteins = sorted(list(set(self.df['protein1'].tolist() + self.df['protein2'].tolist())))

        self.edges_index = []
        self.weights = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Constructing edges index'):
            protein_id1 = self.protein_name_to_id(row['protein1'])
            protein_id2 = self.protein_name_to_id(row['protein2'])
            self.edges_index.append((protein_id1,protein_id2))
            self.edges_index.append((protein_id2,protein_id1))
            if load_weights:
                self.weights.extend([row['weight']] * 2)



    def go_term_to_id(self, go_term):
        return str(int(go_term.split(':')[-1]))
    
    def get_go_terms_and_embeddings(self, protein_semantic_name, load_embeddings=True, name_spaces=['MF','BP','CC']):
        mf_list = []
        mf_emb = []
        if 'MF' in name_spaces:
            try:
                mf_list = list(self.ns2assc['MF'][self.semantic_name_to_sgd_id[protein_semantic_name]])
                if load_embeddings:
                    mf_emb = list(map(lambda x: self.mf_embeddings[self.go_term_to_id(x)], mf_list))
            except:
                mf_list = []
                mf_emb = []
        
        bp_list = []
        bp_emb = []
        if 'BP' in name_spaces:
            try:
                bp_list = list(self.ns2assc['BP'][self.semantic_name_to_sgd_id
                [protein_semantic_name]])
                if load_embeddings:
                    bp_emb = list(map(lambda x: self.bp_embeddings[self.go_term_to_id(x)], bp_list))
            except:
                bp_list = []
                bp_emb = []
        cc_list = []
        cc_emb = []
        if 'CC' in name_spaces:
            try:
                cc_list = list(self.ns2assc['CC'][self.semantic_name_to_sgd_id[protein_semantic_name]])
                if load_embeddings:
                    cc_emb = list(map(lambda x: self.cc_embeddings[self.go_term_to_id(x)], cc_list))
            except:
                cc_list = []
                cc_emb = []
                
        go_terms = mf_list + bp_list + cc_list
        
        go_embeddings = mf_emb + bp_emb + cc_emb if load_embeddings else []

        return go_terms, go_embeddings

    def protein_name_to_id(self, protein_name):
        return self.proteins.index(protein_name)

    def id_to_protein_name(self, id):
        return self.proteins[id]
    
    def get_features(self, type='one_hot', name_spaces=['MF','BP','CC']):
        features = []
        for protein in tqdm(self.proteins, desc=f'loading {type}'):
            go_terms, embeddings = self.get_go_terms_and_embeddings(protein, load_embeddings=self.load_embeddings, name_spaces=name_spaces)

            if type == 'one_hot':
                features.append(go_terms)
            elif type == 'embedding':
                features.append(embeddings)
        if type == 'one_hot':
            self.all_go_terms = sorted(list(set([go_term for protein_go_terms in features for go_term in protein_go_terms])))

            one_hot = np.zeros( (len(self.proteins), len(self.all_go_terms)) )

            for protein_idx, protein_go_terms in enumerate(features):
                for go_term in protein_go_terms:
                    go_term_idx = self.all_go_terms.index(go_term)
                    one_hot[protein_idx, go_term_idx] = 1

            features = one_hot

        return features


    def __repr__(self):
        dataset_name = os.path.basename(self.csv_path).split('.')[0]
        num_proteins = len(self.proteins)
        num_interactions = len(self.df)

        return f'Dataset Loader for {dataset_name}\nnum_proteins:{num_proteins}\nnum_interactions:{num_interactions}'