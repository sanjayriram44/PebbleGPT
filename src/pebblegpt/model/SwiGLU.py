import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2816):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.content = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.swish = F.silu
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.swish(self.gate(x)) * self.content(x))
    
