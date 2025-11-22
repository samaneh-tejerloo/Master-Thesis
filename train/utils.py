import torch
from tqdm import tqdm


def process_features(features, edge_index, edge_weights=None):
    _features = torch.zeros((len(features), 128), dtype=torch.float32)
    for idx, feature in enumerate(features):
        if len(feature) > 0:
            feature = torch.tensor(feature)
            feature = feature.mean(dim=0)
            _features[idx] = feature

    for idx, feature in enumerate(features):
        if len(feature) == 0:
            indices = torch.where(edge_index[0, :] == idx)[0]
            target_nodes = edge_index[:, indices][1, :]
            if edge_weights is not None:
                target_weights = edge_weights[indices]
            else:
                target_weights = torch.ones_like(indices)

            sum_embeddings = torch.zeros(1, 128)
            sum_weights = 0
            for target_node, weight in zip(
                target_nodes.tolist(), target_weights.tolist()
            ):
                feature = _features[target_node]
                if feature.sum() != 0:
                    sum_embeddings += weight * feature
                    sum_weights += weight
            if sum_weights == 0:
                _features[idx] = sum_embeddings
            else:
                _features[idx] = sum_embeddings / sum_weights
    return _features


def calculate_TOM(A):
    # calculating the TOM
    A_hat = torch.maximum(A, A.T)
    A_hat = A_hat.fill_diagonal_(0)
    K = A_hat.sum(dim=1)
    L = A_hat @ A_hat
    Kmin = torch.min(K[:, None], K[None, :])
    numerator = L + A_hat
    denominator = Kmin + 1.0 - A_hat
    TOM = numerator / (denominator + 1e-12)
    TOM = torch.maximum(TOM, TOM.T)
    TOM = TOM.fill_diagonal_(0)
    return TOM


def get_algorithm_complexes(F_out, dataset, threshold=0.3):
    clustering = (F_out > threshold).to(torch.int8)

    algorithm_complexes = []
    for cluser_id in range(clustering.shape[1]):
        indices = torch.where(clustering[:, cluser_id] == 1)[0]
        if len(indices) > 0:
            alg_complex = []
            for protein_idx in indices.tolist():
                protein_name = dataset.id_to_protein_name(protein_idx)
                alg_complex.append(protein_name)
            algorithm_complexes.append(alg_complex)

    print("Number of clusters", len(algorithm_complexes))
    print(
        "Number of clusters with one protein",
        sum([len(c) <= 1 for c in algorithm_complexes]),
    )
    algorithm_complexes = [c for c in algorithm_complexes if len(c) > 1]
    print("Number of algorithm complexes:", len(algorithm_complexes))
    return algorithm_complexes


def calculate_loss_inter_intra(dataset, F_out, threshold, beta=1):
    clusters = get_algorithm_complexes(F_out, dataset, threshold)

    clusters_id = []
    for cluster in clusters:
        cluster_id = []
        for p in cluster:
            cluster_id.append(dataset.protein_name_to_id(p))
        clusters_id.append(cluster_id)

    l_intra = 0
    for cluster_id in tqdm(clusters_id, desc="l_intra"):
        c = len(cluster_id)
        denominator = c * (c - 1)
        sum = 0
        for i in cluster_id:
            for j in cluster_id:
                if i != j:
                    sum += torch.norm(F_out[i] - F_out[j])
        l_intra += sum / denominator

    l_intra /= len(clusters_id)

    l_inter = 0
    for cluster_id in tqdm(clusters_id, desc="l_inter"):
        c = len(cluster_id)
        n = len(dataset.proteins)
        denominator = c * (n - c)
        sum = 0
        for i in cluster_id:
            for j in range(n):
                if j not in cluster_id:
                    sum += torch.norm(F_out[i] - F_out[j])
        l_inter += sum / denominator
    l_inter /= len(clusters_id)

    loss = l_intra + beta * (1 / l_inter)
    return loss
