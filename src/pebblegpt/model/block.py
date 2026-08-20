import torch
import torch.nn as nn

from pebblegpt.model.attention import GQA
from pebblegpt.model.SwiGLU import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 2816,
             num_heads: int = 16, num_kv_heads: int = 4, max_seq_len: int = 2048,
             rope_base: float = 10000.0, norm_eps: float = 1e-6):
        super().__init__()
        self.attn_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = GQA(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.mlp_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x