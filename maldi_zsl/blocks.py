import torch
import torch.nn as nn


class Permute(nn.Module): 
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args)
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim = 64, kernel_size = 5, dropout = 0.2): # a simple residual block
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding = "same"), #channels in, channels out, kernel size
            nn.ReLU(),
            nn.Dropout(dropout),
            Permute(0,2,1),
            nn.LayerNorm(hidden_dim),
            Permute(0,2,1),
            )
        # NOTE: these permutes are necessary here because LayerNorm expects the "hidden_dim" to be the last dim
        # While for Conv1d the "channels" or "hidden_dims" are the second dimension.
        # These permutes basically swap BxCxL to BxLxC for the layernorm, and afterwards swap them back
    def forward(self, x):
        return self.net(x) + x #residual connection
    
class GlobalPool(nn.Module):
    def __init__(self, pooled_axis = 1, mode = "max"):
        super().__init__()
        assert mode in ["max", "mean"], "Only max and mean-pooling are implemented"
        if mode == "max":
            self.op = lambda x: torch.max(x, axis = pooled_axis).values
        elif mode == "mean":
            self.op = lambda x: torch.mean(x, axis = pooled_axis).values
    def forward(self, x):
        return self.op(x)