import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import PredictionHead, TopKPool, TopKUnpool
from model.info_aggregator import WeatherGCNet
from model.lightning_base import LightningBase

class WeatherGC_UNet(LightningBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)

        self.K = hparams.pool_ratio
        self.sigma = hparams.sigma

        # Normalization of the input data
        self.data_bn = nn.BatchNorm1d(4*5) # channels (C) * vertices (V)

        self.st_agg_in = WeatherGCNet(hparams)

        self.pool = TopKPool(k=self.K, input_dim=4, p=0.1)
        self.st_agg1 = WeatherGCNet(hparams)
        self.unpool = TopKUnpool()
        self.st_agg2 = WeatherGCNet(hparams)
        self.pred_head = PredictionHead(4, hparams.num_timesteps, hparams.num_nodes, 3)

    def compute_adjacency(self, X, sigma=0.35):
        B, C, T, V = X.size()

        # [B, C, V]
        X_avg = X.mean(dim=2)

        normed = F.normalize(X_avg, dim=1)
        # [B, V, C]
        normed = normed.permute(0, 2, 1)

        # [B, V, 1, C]
        X_i = normed.unsqueeze(2)
        # [B, 1, V, C]
        X_j = normed.unsqueeze(1)

        # [B, V, V]
        dist_sqd = ((X_i - X_j)**2).sum(dim=-1)

        adj = torch.exp(-dist_sqd / (2 * sigma**2))
        return adj

    
    def forward(self, X):
        B, C, T, V = X.size()

        # Normalization
        X = X.permute(0, 3, 1, 2).contiguous().view(B, V*C, T)
        X = self.data_bn(X)
        X = X.view(B, V, C, T).permute(0, 2, 3, 1).contiguous().view(B, C, T, V)

        # Calculate initial A with input features
        A = self.compute_adjacency(X, self.sigma)

        X_in = self.st_agg_in(X, A)

        # Encoder
        A_pooled, X_pooled, selected_idx = self.pool(X_in, A)
        X1 = self.st_agg1(X_pooled, A_pooled)

        # Decoder
        X_unpooled = self.unpool(X1, X_in, A, selected_idx)

        # Additive fusion
        X_fused = torch.add(X_unpooled, X_in)
        X2 = self.st_agg2(X_fused, A)

        # [B, C*T*V]
        X2 = X2.reshape(B, -1)

        Y_hat = self.pred_head(X2)
        return Y_hat, A