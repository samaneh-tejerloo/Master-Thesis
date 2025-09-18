# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 21:31:27 2024

@author: tejer
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

colins_df = pd.read_csv('krogan-extended.csv', index_col=0)

number_of_interactions = len(colins_df)

proteins = set(colins_df['protein1'].to_list()+colins_df['protein2'].to_list())
number_of_proteins = len(proteins)


G = nx.Graph()

G.add_nodes_from(proteins)

p1 = colins_df['protein1'].to_list()
p2 = colins_df['protein2'].to_list()
edges = list(zip(p1,p2))

G.add_edges_from(edges)

plt.figure(figsize=(20,20))
nx.draw(G, with_labels=False)

average_clustering_coefficient = nx.average_clustering(G)

sum_n = 0
for node in G.nodes:
    number_of_neighbors = len(G.adj[node])
    sum_n += number_of_neighbors
average_number_of_neighbor = sum_n/number_of_proteins  


print(f'Number of Proteins: {number_of_proteins}')
print(f'Number of Interactions: {number_of_interactions}')
print(f'Average Clustering Coefficient: {average_clustering_coefficient}')
print(f'Average Number of Neighbors: {average_number_of_neighbor}')
