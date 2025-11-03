import torch

def process_features(features, edge_index, edge_weights=None):
    _features = torch.zeros((len(features), 128), dtype=torch.float32)
    for idx, feature in enumerate(features):
        if len(feature) > 0:
            feature = torch.tensor(feature)
            feature = feature.mean(dim=0)
            _features[idx] = feature

    for idx, feature in enumerate(features):
        if len(feature) == 0:
            indices = torch.where(edge_index[0,:]==idx)[0]
            target_nodes = edge_index[:, indices][1,:]
            if edge_weights is not None:
                target_weights = edge_weights[indices]
            else:
                target_weights = torch.ones_like(indices)
            
            sum_embeddings = torch.zeros(1,128)
            sum_weights = 0
            for target_node, weight in zip(target_nodes.tolist(), target_weights.tolist()):
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