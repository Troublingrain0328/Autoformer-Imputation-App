import torch
import torch.nn as nn
from layers.Autoformer_EncDec import Encoder, EncoderLayer, series_decomp, my_Layernorm
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Embed import DataEmbedding_wo_pos

class AutoformerNet(nn.Module):
    def __init__(self, configs):
        super(AutoformerNet, self).__init__()
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                moving_avg=configs.moving_avg,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=my_Layernorm(configs.d_model))
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

def load_model_and_config():
    class Config:
        enc_in = 7
        d_model = 64
        embed = 'timeF'
        freq = 'h'
        dropout = 0.1
        factor = 3
        output_attention = False
        n_heads = 4
        d_ff = 128
        moving_avg = 25
        activation = 'gelu'
        e_layers = 2
        c_out = 7
        seq_len = 96

    config = Config()
    model = AutoformerNet(config)
    # model.load_state_dict(torch.load("checkpoints/autoformer.pth"))  # 若有预训练模型
    return model, config

