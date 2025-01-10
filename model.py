import torch.nn as nn
from transformer.Transformer import Transformer
from transformer_config import TransformerConfig

def build_model(cfg: TransformerConfig, src_vocab_size:int, tgt_vocab_size:int) -> Transformer:

        transformer = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            cfg.src_seq_len,
            cfg.tgt_seq_len,
            cfg.d_model,
            cfg.num_layers,
            cfg.d_ff,
            cfg.h,
            cfg.dropout
        )

        # initialize parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return transformer