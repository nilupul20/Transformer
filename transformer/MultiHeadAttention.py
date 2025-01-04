import math
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model%h == 0, f"d_model {d_model} is not divisible by h {h}"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask==0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # (batch_size, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value) , attention_scores



    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, h, d_k) --> # (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k )

        return self.w_o(x) # (batch_size, seq_len, d_model)
