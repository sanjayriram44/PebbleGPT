"""Metrics logging: console, JSONL, and optionally Weights & Biases.

The Playbook's hardest-won lesson: loss can look completely normal while the
model is silently broken. Their TP bug survived to 1T tokens because the loss
curve was fine — only downstream evals caught it. So log more than loss:
gradient norm (spikes here often precede loss spikes), throughput, and MFU.
"""

import json
import time
from pathlib import Path

import torch


class Logger:
    def __init__(self,
                 run_name: str,
                 log_dir: Path = Path("logs"),
                 use_wandb: bool = False,
                 wandb_project: str = "PebbleGPT",
                 wandb_entity: str | None = None,
                 config: dict | None = None,
                 log_every: int = 10):
        self.run_name = run_name
        self.log_every = log_every
        self.t_start = time.time()
        self.t_last = self.t_start

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(log_dir / f"{run_name}.jsonl", "a")

        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config=config,
                )
                self.wandb = wandb
            except Exception as e:
                # Never let a tracking outage kill a long run.
                print(f"wandb unavailable, continuing without it: {e}")

    def log(self, step: int, metrics: dict, force: bool = False) -> None:
        if not force and step % self.log_every != 0:
            return

        now = time.time()
        metrics = {**metrics,
                   "step": step,
                   "elapsed_s": round(now - self.t_start, 1)}

        self.jsonl.write(json.dumps(metrics, default=float) + "\n")
        self.jsonl.flush()

        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

        parts = [f"step {step:>7,}"]
        for k in ("loss", "lr", "grad_norm", "tok_per_s", "mfu"):
            if k not in metrics:
                continue
            v = metrics[k]
            if k == "loss":
                parts.append(f"loss {v:.4f}")
            elif k == "lr":
                parts.append(f"lr {v:.2e}")
            elif k == "grad_norm":
                parts.append(f"gnorm {v:.2f}")
            elif k == "tok_per_s":
                parts.append(f"{v:,.0f} tok/s")
            elif k == "mfu":
                parts.append(f"mfu {v * 100:.1f}%")
        print("  ".join(parts), flush=True)

        self.t_last = now

    def close(self) -> None:
        self.jsonl.close()
        if self.wandb is not None:
            self.wandb.finish()


def compute_mfu(tokens_per_sec: float,
                n_params: int,
                peak_flops: float) -> float:
    """Model FLOPs Utilization.

    Uses the 6*N*D approximation: ~6 FLOPs per parameter per token.

    Below ~20% means something is wrong — usually torch.compile not engaging
    or SDPA falling back to the math path instead of FlashAttention.
    """
    return (tokens_per_sec * 6 * n_params) / peak_flops


PEAK_FLOPS_BF16 = {
    "rtx4090": 83e12,
    "a100": 312e12,
    "h100": 989e12,
    "h200": 989e12,
}


def guess_peak_flops(default: float = 312e12) -> float:
    """Best-effort peak FLOPS from the device name."""
    if not torch.cuda.is_available():
        return default
    name = torch.cuda.get_device_name(0).lower()
    for key, flops in PEAK_FLOPS_BF16.items():
        if key.replace("rtx", "") in name.replace(" ", ""):
            return flops
    print(f"unknown GPU '{name}', assuming {default:.0e} FLOPS for MFU")
    return default