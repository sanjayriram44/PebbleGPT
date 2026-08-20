import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pebblegpt.model.block import TransformerBlock


class PebbleGPT(nn.Module):
    def __init__(self, vocab_size: int = 49152, hidden_size: int = 1024, num_hidden_layers: int = 24,
             num_heads: int = 16, num_kv_heads: int = 4, intermediate_size: int = 2816,
             max_seq_len: int = 2048, rope_base: float = 10000.0, norm_eps: float = 1e-6):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_size=intermediate_size,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                norm_eps=norm_eps)
            for _ in range(num_hidden_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.proj_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        self.proj_head.weight = self.token_embedding.weight
        
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.token_embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.proj_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
        
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n