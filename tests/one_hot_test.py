#%%
from dataset import PPIDataLoadingUtil
#%%
ppi_dataset = PPIDataLoadingUtil('datasets/colins.csv',load_embeddings=True)
# %%
one_hot_features_BP = ppi_dataset.get_features(type='one_hot', name_spaces=['BP'])
one_hot_features_MF = ppi_dataset.get_features(type='one_hot', name_spaces=['MF'])
one_hot_features_CC = ppi_dataset.get_features(type='one_hot', name_spaces=['CC'])
one_hot_features_ALL = ppi_dataset.get_features(type='one_hot')
# %%
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_embeddings_BP = tsne.fit_transform(one_hot_features_BP)
tsne_embeddings_MF = tsne.fit_transform(one_hot_features_MF)
tsne_embeddings_CC = tsne.fit_transform(one_hot_features_CC)
tsne_embeddings_ALL = tsne.fit_transform(one_hot_features_ALL)
# %%
import matplotlib.pyplot as plt
from evaluate import Evaluation
evaluate = Evaluation('datasets/sgd.txt', ppi_dataset)
evaluate.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
# %%
complexes = evaluate.filtered_complexes
# %%
for idx, complex in enumerate(complexes):
    colors = ['green' if protein in complex else 'yellow' for protein in ppi_dataset.proteins]
    plt.figure(figsize=(15,15))
    plt.title(f'# Proteins: {len(complex)}')
    plt.axis('off')
    plt.subplot(2,2,1)
    plt.scatter(tsne_embeddings_BP[:,0], tsne_embeddings_BP[:,1], s=5, c=colors, alpha=0.5)
    plt.title('BP')

    plt.subplot(2,2,2)
    plt.scatter(tsne_embeddings_MF[:,0], tsne_embeddings_MF[:,1], s=5, c=colors, alpha=0.5)
    plt.title('MF')

    plt.subplot(2,2,3)
    plt.scatter(tsne_embeddings_CC[:,0], tsne_embeddings_CC[:,1], s=5, c=colors, alpha=0.5)
    plt.title('CC')

    plt.subplot(2,2,4)
    plt.scatter(tsne_embeddings_ALL[:,0], tsne_embeddings_ALL[:,1], s=5, c=colors, alpha=0.5)
    plt.title('ALL')
    plt.savefig(f'temp/one_hot/{idx}.jpg')
# %%
