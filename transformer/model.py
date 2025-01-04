import torch.nn as nn
from transformer.Transformer import Transformer

def build_model(cfg: dict) -> Transformer:
        src_vocab_size = cfg['src_vocab_size']
        tgt_vocab_size = cfg['tgt_vocab_size']

        src_seq_len = cfg['src_seq_len']
        tgt_seq_len = cfg['tgt_seq_len']

        d_model = cfg.get('d_model', 512)
        num_layers = cfg.get('num_layers', 6)
        d_ff = cfg.get('d_ff', 2048)
        h = cfg.get('h', 8)
        dropout = cfg.get('dropout', 0.1)

        transformer = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            src_seq_len,
            tgt_seq_len,
            d_model,
            num_layers,
            d_ff,
            h,
            dropout
        )

        # initialize parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return transformer