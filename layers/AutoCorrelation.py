import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # 快速傅里叶变换实现时序自注意力
        B, L, D = queries.shape
        res = torch.fft.ifft(torch.fft.fft(queries, dim=1) * torch.fft.fft(keys, dim=1).conj(), dim=1).real
        weights = res / res.sum(dim=-1, keepdim=True)
        if self.mask_flag:
            weights = self.dropout(weights)
        out = torch.einsum("blh,bld->bhd", weights, values)
        if self.output_attention:
            return out, weights
        else:
            return out, None

class AutoCorrelationLayer(nn.Module):
    def __init__(self, autocorrelation, d_model, n_heads):
        super(AutoCorrelationLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.auto = autocorrelation

    def forward(self, queries, keys, values, attn_mask):
        B, L, D = queries.shape
        queries = self.q_proj(queries).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        keys = self.k_proj(keys).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        values = self.v_proj(values).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        out, attn = self.auto(queries, keys, values, attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out), attn

