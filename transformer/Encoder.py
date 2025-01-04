import torch.nn as nn
from transformer.LayerNorm import LayerNorm
from transformer.EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, num_layers:int, d_model:int, h:int, d_ff:int, dropout:float):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)