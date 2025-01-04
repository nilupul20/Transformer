import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x): # (batch_size, seq_len, d_model)
        x = torch.relu(self.lin1(x)) # (batch_size, seq_len, d_ff)
        x = self.dropout(x)
        return self.lin2(x) # (batch_size, seq_len, d_model)