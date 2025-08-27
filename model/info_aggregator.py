import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((input_dim, output_dim)))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

        # Residual
        if input_dim != output_dim:
            self.down = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 1),
                nn.BatchNorm2d(output_dim)
            )
        else:
            self.down = lambda x: x

    def forward(self, X, A):
        B, C, T, V = X.size()
        # [B, T, V, C]
        X = X.permute(0, 2, 3, 1)

        # Add self-loops to A
        I = torch.eye(V, device=A.device).unsqueeze(0)
        A_tilde = A + I

        # [B, V]
        D_tilde = torch.sum(A_tilde, dim=2)
        D_tilde_sqrt = torch.pow(D_tilde, -0.5)
        # Avoid infs
        D_tilde_sqrt[torch.isinf(D_tilde_sqrt)] = 0.0
        
        # [B, V, V]
        D_tilde_sqrt_mat = torch.diag_embed(D_tilde_sqrt)

        # Symmetric normalization
        # [B, V, V]
        A_norm = D_tilde_sqrt_mat @ A_tilde @ D_tilde_sqrt_mat

        # Message Passing
        # [B, 1, V, V] @ [B, T, V, C] -> [B, T, V, C]
        y = torch.matmul(A_norm.unsqueeze(1), X)
        # Convolution
        # [B, T, V, C] @ [C, output_dim] -> [B, T, V, output_dim]
        y = torch.matmul(y, self.weight)
        y = y + self.bias

        # [B, C, T, V]
        y = y.permute(0, 3, 1, 2)
        y = self.bn(y)
        residual = self.down(X.permute(0, 3, 1, 2))
        y += residual
        y = self.relu(y)

        return y

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class TemporalMessagePassing(nn.Module):
    """
    Based on TCN_unit of WeatherGCNet: https://github.com/tstanczyk95/WeatherGCNet/blob/main/DK%20wind%20speed%20forecasting/WeatherGCNet%20with%20gamma/model_dk.py
    """
    def __init__(self, hparams, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.hparams = hparams

        pad = int((kernel_size-1) / 2)

        # Kernel_size of (k, 1) enables a 1x1 temporal convolution over all V nodes in parallel
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(pad, 0))
        self.norm = nn.BatchNorm2d(out_channels)

        conv_init(self.conv)
        bn_init(self.norm, 1)

    def forward(self, X):
        # Input: [B, C, T, V]
        out = self.conv(X)
        out = self.norm(out)

        return out

class GCN_ST_Unit(nn.Module):
    def __init__(self, hparams, in_channels, out_channels, stride=1, residual=True):
        super().__init__()

        self.dropout = nn.Dropout(p=0.05)

        self.spatial_proc = GCN(in_channels, out_channels)
        self.temporal_proc = TemporalMessagePassing(hparams, in_channels=out_channels, out_channels=out_channels)

        self.act = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels==out_channels) and (stride==1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalMessagePassing(hparams, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, X, A):
        B, C, T, V = X.size()

        X = self.dropout(X)

        # [B, C_out, T, V]
        X = self.spatial_proc(X, A) + self.residual(X)
        # [B, C_out, T, V]
        X = self.temporal_proc(X)
        out = self.act(X)
        return out
    
class WeatherGCNet(nn.Module):
    """
    Based on https://github.com/tstanczyk95/WeatherGCNet/blob/main/DK%20wind%20speed%20forecasting/WeatherGCNet%20with%20gamma/model_dk.py
    
    """
    def __init__(self, hparams):
        super().__init__()

        self.l1 = GCN_ST_Unit(hparams, in_channels=4, out_channels=8)
        self.l2 = GCN_ST_Unit(hparams, in_channels=8, out_channels=16)
        self.l3 = GCN_ST_Unit(hparams, in_channels=16, out_channels=32)

        # Reduce channel count back to 4
        self.conv_reduce_channels = TemporalMessagePassing(hparams, 32, 4)

    def forward(self, X, A):
        B, C, T, V = X.size()

        X = self.l1(X, A)
        X = self.l2(X, A)
        X = self.l3(X, A)

        out = self.conv_reduce_channels(X)
        return out