#%%
import numpy as np
from dataset import PPIDataLoadingUtil
from evaluate import Evaluation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from AdaPPI.evaluator.compare_performance import get_score
#%%
ppi_data_loader = PPIDataLoadingUtil('datasets/tadw-sc/collins_2007/colins2007.csv', load_weights=True)
# %%
n = len(ppi_data_loader.proteins)
A = np.zeros((n,n))
edge_indices = np.array(ppi_data_loader.edges_index)
# mode = 'non-weighted'
mode = 'weighted'
if mode == 'weighted':
    for edge, weight in zip(edge_indices, ppi_data_loader.weights):
        A[edge[0],edge[1]] = weight
else:
    A[edge_indices[:,0],edge_indices[:,1]] = 1

#%%
def column_normalize(M):
    col_sums = M.sum(axis=0, keepdims=True)
    # avoid division by zero:
    col_sums[col_sums == 0] = 1.0
    return M / col_sums

def inflate(M, r):
    M = np.power(M, r)
    return column_normalize(M)

def mcl(A, expansion=2, inflation=2.0, loop_value=1.0,
        prune_threshold=1e-5, max_iters=200, tol=1e-6):
    A = A.astype(float).copy()
    np.fill_diagonal(A, loop_value)            # add self-loops
    M = column_normalize(A)                    # initial Markov matrix

    for it in range(max_iters):
        M_prev = M.copy()

        # Expansion: matrix power
        M = np.linalg.matrix_power(M, expansion)

        # Inflation: elementwise power then normalize
        M = inflate(M, inflation)

        # Prune tiny entries to keep sparsity
        M[M < prune_threshold] = 0.0
        M = column_normalize(M)

        # Convergence check
        if np.linalg.norm(M - M_prev) < tol:
            print('Threshold stop')
            break
    n = M.shape[0]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    nz = (M > 0)
    for i in range(n):
        for j in range(n):
            if nz[i,j] or nz[j,i]:
                union(i,j)

    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values()), M
#%%
clusters, M = mcl(A, expansion=2,inflation=2)
print('Number of clusters', len(clusters))
print('Number of clusters with one protein', sum([len(c) <= 1 for c in clusters]))

# evaluate = Evaluation('datasets/golden standard/sgd.txt', ppi_data_loader)
evaluate = Evaluation('datasets/golden standard/ada_ppi.txt', ppi_data_loader)
evaluate.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
complexes = evaluate.filtered_complexes
algorithm_complexes = []
for cluster in clusters:
    alg_complex = [ppi_data_loader.id_to_protein_name(protein_idx) for protein_idx in cluster]
    if len(alg_complex) > 1:
        algorithm_complexes.append(alg_complex)
print('Number of algorithm complexes:', len(algorithm_complexes))
# algorithm_complexes = []
# for protein in ppi_data_loader.proteins:
#     algorithm_complexes.append([protein])

# evaluate.evalute(algorithm_complexes)
get_score(algorithm_complexes, complexes)
#%%
plt.hist([len(complex) for complex in complexes], bins=1000)
# %%
tsne = TSNE(n_components=2, perplexity=10)
A_tsne = tsne.fit_transform(M)
evaluate = Evaluation('datasets/golden standard/ada_ppi.txt', ppi_data_loader)
evaluate.filter_reference_complex(filtering_method='just_keep_dataset_proteins')
complexes = evaluate.filtered_complexes
# %%
for idx, comp in enumerate(complexes):
    in_comp = np.array([p in comp for p in ppi_data_loader.proteins])
    # Red points
    plt.scatter(A_tsne[in_comp,0], A_tsne[in_comp,1],
                c='red', zorder=2)
    # Blue points
    plt.scatter(A_tsne[~in_comp,0], A_tsne[~in_comp,1],
                c='blue', zorder=1)
    plt.show()
    if idx == 10:
        break
# %%
