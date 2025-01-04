import torch.nn as nn
from transformer.LayerNorm import LayerNorm
from transformer.DecoderLayer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, num_layers:int, d_model:int, h:int, d_ff:int, dropout:float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)