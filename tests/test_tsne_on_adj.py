#%%
from dataset import PPIDataLoadingUtil
from models import SimpleGCN
from torch_geometric.data import Data
import torch
import nocd
from tqdm import tqdm
from evaluate import Evaluation
from constants import SGD_GOLD_STANDARD_PATH
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from evaluate import Evaluation
#%%
ppi_data_loader = PPIDataLoadingUtil('datasets/colins.csv')
# %%
n = len(ppi_data_loader.proteins)
A = np.zeros((n,n))
edge_indices = np.array(ppi_data_loader.edges_index)
A[edge_indices[:,0],edge_indices[:,1]] = 1
# %%
tsne = TSNE(n_components=2, perplexity=10)
A_tsne = tsne.fit_transform(A)
evaluate = Evaluation('datasets/sgd.txt', ppi_data_loader)
evaluate.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
complexes = evaluate.filtered_complexes
#%%
for idx, complex in enumerate(complexes):
    plt.figure(figsize=(10,10))
    colors = ['red' if protein in complex else 'blue' for protein in ppi_data_loader.proteins]
    s = [30 if protein in complex else 5 for protein in ppi_data_loader.proteins]

    plt.scatter(A_tsne[:,0], A_tsne[:,1], s=s, c=colors, alpha=0.5)



    for edge in edge_indices:
        p1_id = edge[0]
        p2_id = edge[1]

        p1_name = ppi_data_loader.id_to_protein_name(p1_id)
        p2_name = ppi_data_loader.id_to_protein_name(p2_id)

        color = (np.random.randint(0,256, 3) / 255).tolist()
        if p1_name in complex:
            if p2_name in complex:
                p1 = A_tsne[p1_id].reshape(1,-1)
                p2 = A_tsne[p2_id].reshape(1,-1)
                points = np.concat([p1,p2])
                plt.plot(points[:,0], points[:,1], c=color,alpha=0.3, zorder=-1)
            else:
                for edge_prime in edge_indices[edge_indices[:,0]== p2_id]:

                    p3_id = edge_prime[1]
                    p3_name = ppi_data_loader.id_to_protein_name(p3_id)

                    if p3_id == p1_id:
                        continue
                    if p3_name in complex:
                        p1 = A_tsne[p1_id].reshape(1,-1)
                        p2 = A_tsne[p2_id].reshape(1,-1)
                        p3 = A_tsne[p3_id].reshape(1,-1)
                        points = np.concat([p1,p2,p3])
                        print()
                        plt.plot(points[:,0], points[:,1], c=color,alpha=0.3, zorder=-1)

    plt.show()
    if idx == 2:
        break
#%%