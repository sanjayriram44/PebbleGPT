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
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base

        self.W_Q = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        # RoPE tables are derived from rope_base and head_dim on first use.
        # Deliberately NOT registered as buffers: they aren't in the state
        # dict, and from_pretrained zero-fills or skips non-persistent buffers,
        # which silently turns RoPE into the identity.
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def _get_rope(self, seq_len: int, device, dtype):
        """cos/sin tables covering [0, seq_len), cached and grown as needed."""
        cache = self._rope_cache
        if (cache is not None
                and cache[0].size(0) >= seq_len
                and cache[0].device == device
                and cache[0].dtype == dtype):
            return cache[0][:seq_len], cache[1][:seq_len]

        n = max(seq_len, self.max_seq_len)
        inv_freq = 1.0 / (self.rope_base ** (
            torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
            / self.head_dim))
        positions = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
        self._rope_cache = (cos, sin)
        return cos[:seq_len], sin[:seq_len]

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
        cos_all, sin_all = self._get_rope(past_len + T, Q.device, Q.dtype)
        cos = cos_all[past_len:past_len + T].unsqueeze(0).unsqueeze(0)
        sin = sin_all[past_len:past_len + T].unsqueeze(0).unsqueeze(0)
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