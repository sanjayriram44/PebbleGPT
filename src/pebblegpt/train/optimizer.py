"""AdamW with weight-decay exclusions.

Hyperparameters are the field standard, reused unchanged across Llama 1/2/3
and DeepSeek V1/V2/V3 (124M to 671B params):
    beta1=0.9, beta2=0.95, weight_decay=0.1, grad_clip=1.0

Embeddings and norm gains are excluded from weight decay. OLMo 2 found this
improves stability: decay shrinks embedding norms over training, and since
LayerNorm's Jacobian is inversely proportional to input norm, that produces
larger gradients in early layers (Takase et al., 2025).
"""

import inspect

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module,
                    lr: float = 5e-4,
                    weight_decay: float = 0.1,
                    beta1: float = 0.9,
                    beta2: float = 0.95,
                    eps: float = 1e-8,
                    device_type: str = "cuda") -> torch.optim.AdamW:
    """AdamW with two param groups: decay and no-decay."""
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude embeddings, norm gains, and any biases. 2-D tensors are
        # weight matrices (decay); 1-D are norms/biases (no decay).
        if "embedding" in name.lower() or param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    print(f"optimizer: {len(decay)} tensors ({n_decay:,} params) with decay, "
          f"{len(no_decay)} tensors ({n_no_decay:,} params) without")

    # Fused AdamW is meaningfully faster on CUDA when available.
    fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra = {"fused": True} if (fused_ok and device_type == "cuda") else {}

    return torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2), eps=eps, **extra)