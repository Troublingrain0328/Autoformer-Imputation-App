import torch
import torch.nn as nn

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.temporal_embedding = nn.Linear(4, d_model)  # month, day, weekday, hour
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
