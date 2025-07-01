import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelationLayer

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        moving_mean = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        res = x - moving_mean
        return res, moving_mean

class my_Layernorm(nn.Module):
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layer = nn.LayerNorm(channels)

    def forward(self, x):
        return self.layer(x)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0.1, activation="relu", moving_avg=25):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.decomp = series_decomp(moving_avg)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        x, _ = self.decomp(x)
        x = x + self.dropout(self.ff(x))
        x = self.norm2(x)
        return x, attn

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

