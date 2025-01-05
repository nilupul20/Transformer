import torch.nn as nn
from transformer.Encoder import Encoder
from transformer.Decoder import Decoder
from transformer.Embedding import InputEmbeddings
from transformer.PositionalEncoding import PositionalEncoding
from transformer.ProjectionLayer import ProjectionLayer


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            src_seq_len: int,
            tgt_seq_len: int,
            d_model:int,
            num_layers:int,
            d_ff: int,
            h: int,
            dropout: float
        ):
        super().__init__()

        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

        self.src_pe = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pe = PositionalEncoding(d_model, tgt_seq_len, dropout)

        self.encoder = Encoder(num_layers, d_model, h, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, h, d_ff, dropout)

        self.projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pe(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pe(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
