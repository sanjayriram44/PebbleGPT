import torch
import torch.nn as nn
import torch.nn.functional as F


class GQA(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, num_kv_heads: int = 4,
                 max_seq_len: int = 2048, rope_base: float = 10000.0,
                 layer_idx: int = 0):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.W_Q = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        cos, sin = self._precompute_rope(self.head_dim, max_seq_len, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _precompute_rope(head_dim, max_seq_len, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def _apply_rope(cls, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (cls._rotate_half(x) * sin)

    def forward(self, x: torch.Tensor, past_key_values=None, past_len: int = 0):
        """past_key_values is a transformers Cache object, or None.

        RoPE is applied before caching, so cached keys already carry their
        position — new tokens must be offset by past_len.
        """
        B, T, _ = x.shape

        # 1. Projections
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # 2. Reshape into heads: [B, T, H*d_h] -> [B, H, T, d_h]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE — these tokens sit at positions [past_len, past_len + T)
        cos = self.rope_cos[past_len:past_len + T].unsqueeze(0).unsqueeze(0).to(Q.dtype)
        sin = self.rope_sin[past_len:past_len + T].unsqueeze(0).unsqueeze(0).to(Q.dtype)
        Q = self._apply_rope(Q, cos, sin)
        K = self._apply_rope(K, cos, sin)

        # 4. Cache.update appends and returns the full accumulated K/V
        if past_key_values is not None:
            K, V = past_key_values.update(K, V, self.layer_idx)

        # 5. Expand KV heads to match Q heads (GQA)
        repeats = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeats, dim=1)
        V = V.repeat_interleave(repeats, dim=1)

        # 6. Attention. With a cache, T == 1 while K spans the full context —
        #    is_causal on a non-square score matrix would mask incorrectly.
        #    A single query legitimately attends to everything cached.
        O = F.scaled_dot_product_attention(Q, K, V, is_causal=(T > 1))

        # 7. Merge heads and project out
        O = O.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.W_O(O)