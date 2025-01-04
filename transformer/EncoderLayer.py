import torch.nn as nn
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.FeedForward import FeedForward
from transformer.ResidualConnection import ResidualConnection

class EncoderLayer(nn.Module):
    def __init__(self, d_model:int, h:int, d_ff:int, dropout:float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x