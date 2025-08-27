import torch
import torch.nn as nn
import numpy as np

class TopKPool(nn.Module):
    """
    Based on 'Graph-UNets', https://github.com/HongyangGao/Graph-U-Nets/blob/master/src/utils/ops.py
    """
    def __init__(self, k, input_dim, p=0):
        super().__init__()

        # k is the pooling ratio
        self.k = k
        self.sigmoid = nn.Sigmoid()
        # The scoring function
        self.proj = nn.Linear(input_dim, 1)
        self.drop = nn.Dropout(p=p) if p>0 else nn.Identity()


    def forward(self, X, A):
        B, C, T, V = X.size()

        # [B, C, V]
        X_avg = X.mean(dim=2)
        # Apply dropout prior to projection
        Z = self.drop(X_avg)
        # [B, V, C]
        Z = Z.permute(0, 2, 1)
        # [B, V]
        score_logits = self.proj(Z).squeeze()
        scores = self.sigmoid(score_logits)
        return top_k_graph(scores, A, X, self.k)

def top_k_graph(scores, A, X, k):
    B, C, T, V = X.size()

    K = int(np.floor(k*V))
    # [B, K], [B, K]
    values, idx = torch.topk(input=scores, k=max(2, K), sorted=True)
    idx, _ = torch.sort(idx, dim=-1, descending=False)

    if len(idx.size()) < 2:
        idx = idx.unsqueeze(0)
        values = values.unsqueeze(0)

    # [B, 1, 1, K] -> [B, C, T, K]
    idx_expanded = idx.unsqueeze(1).unsqueeze(1).expand(B, C, T, K)

    # [B, C, T, K]
    gathered_features = torch.gather(X, dim=-1, index=idx_expanded)
    
    # Scale features by values to enable gradient flow
    # [B, C, T, K]
    values = values.unsqueeze(1).unsqueeze(1).expand(B, C, T, K)
    scaled_features = torch.mul(gathered_features, values)

    # Downsample adjacency matrix A
    # Gather rows -> [B, K, V]
    idx_rows = idx.unsqueeze(-1).expand(B, K, V)
    A_rows = torch.gather(A, dim=1, index=idx_rows)
    
    # Gather columns -> [B, K, K]
    idx_columns = idx.unsqueeze(1).expand(B, K, K)
    A_pooled = torch.gather(A_rows, dim=2, index=idx_columns)

    return A_pooled, scaled_features, idx

class TopKUnpool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X_pooled, X_original, A, idx):
        B, C, T, V = X_original.size()
        _, _, _, K = X_pooled.size()
        new_X = torch.zeros_like(X_original)

        # [B, C, T, K]
        idx_expanded = idx.unsqueeze(1).unsqueeze(2).expand(B, C, T, K)

        new_X = new_X.scatter(dim=3, index=idx_expanded, src=X_pooled)

        return new_X

class PredictionHead(nn.Module):
    def __init__(self, channels, num_timesteps, num_nodes, num_targets):
        super().__init__()

        self.fc = nn.Linear(in_features=channels*num_timesteps*num_nodes, out_features=num_targets)

    def forward(self, X):
        return self.fc(X)
