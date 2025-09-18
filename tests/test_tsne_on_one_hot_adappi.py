#%%
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from evaluate import Evaluation
from dataset import PPIDataLoadingUtil
#%%
go_information = {}
with open('AdaPPI/dataset/COLLINS/collins_go_information.txt') as f:
    for line in f.readlines():
        parts = line.split()
        go_information[parts[0]] = parts[1:]
# %%
all_go_terms = sorted(list(set([go_term for protein_name, go_terms in go_information.items() for go_term in go_terms])))
proteins = sorted(list(go_information.keys()))
# %%
features = np.zeros((len(proteins),len(all_go_terms)))
features.shape
# %%
for protein_idx, protein in enumerate(proteins):
    go_terms = go_information[protein]
    for go_term in go_terms:
        go_term_idx = all_go_terms.index(go_term)
        features[protein_idx, go_term_idx] = 1
# %%
tsne = TSNE(n_components=2)
tsne_embeddings = tsne.fit_transform(features)
# %%
ppi_dataset = PPIDataLoadingUtil('datasets/tadw-sc/collins_2007/colins2007.csv',load_embeddings=True)
evaluate = Evaluation('datasets/golden standard/ada_ppi.txt', ppi_dataset)
evaluate.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
complexes = evaluate.filtered_complexes
#%%
for idx, complex in enumerate(complexes):
    colors = ['red' if protein in complex else 'blue' for protein in proteins]
    size = [40 if protein in complex else 10 for protein in proteins]
    plt.figure(figsize=(10,10))
    plt.scatter(tsne_embeddings[:,0], tsne_embeddings[:,1], alpha=0.5, s=size, c=colors)
    plt.show()
    if idx == 10:
        break
# %%
from AdaPPI.evaluator.compare_performance import get_score
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
# %%
kmeans = KMeans(n_clusters=554)
kmeans = kmeans.fit(features)
cluster_ids = kmeans.labels_
#%%
dbscan = DBSCAN(eps=0.1)
dbscan =dbscan.fit(features)
cluster_ids = dbscan.labels_
#%%
spectral = SpectralClustering(n_clusters=500)
spectral = spectral.fit(features)
cluster_ids = spectral.labels_
# %%
clusters = {}
for protein, cluster_id in zip(proteins, cluster_ids):
    cluster_id = cluster_id.item()
    if cluster_id == -1:
        continue
    if cluster_id in clusters:
        clusters[cluster_id].append(protein)
    else:
        clusters[cluster_id] = [protein]
# %%
algorithm_complexes = [v for k,v in clusters.items() if len(v) > 1]
# %%
get_score(algorithm_complexes, complexes)
# %%