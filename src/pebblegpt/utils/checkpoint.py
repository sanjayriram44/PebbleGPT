"""Atomic checkpoint save/load with full resume state.

Writes to a temp file then os.replace() — an atomic rename on POSIX, so a
crash mid-write can never leave a corrupt checkpoint at the real path.

Saves model + optimizer + scheduler + RNG state. RNG matters: without it,
resuming produces a different data order and different dropout draws than
an uninterrupted run would have.

Test resume by killing the process, not by trusting that it works. The
Playbook flags this hardest, and John's 350M writeup found three separate
bugs here (non-atomic saves, no corrupt-checkpoint fallback, missing RNG
restore) — each of which would have silently ruined a long run.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: Path,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler,
                    step: int,
                    tokens_seen: int,
                    config: dict | None = None,
                    keep_last: int = 3) -> Path:
    """Atomically save full training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": {
            "python": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
        },
        "config": config or {},
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic on POSIX

    _prune_old(path.parent, keep_last)
    return path


def load_checkpoint(path: Path,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler=None,
                    device: str = "cpu",
                    restore_rng: bool = True) -> dict:
    """Restore training state. Returns metadata (step, tokens_seen, config)."""
    state = torch.load(Path(path), map_location=device, weights_only=False)

    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])

    if restore_rng and "rng" in state:
        rng = state["rng"]
        np.random.set_state(rng["python"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])

    return {
        "step": state["step"],
        "tokens_seen": state["tokens_seen"],
        "config": state.get("config", {}),
    }


def find_latest(ckpt_dir: Path) -> Path | None:
    """Most recent valid checkpoint, skipping any that fail to load."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    paths = sorted(ckpt_dir.glob("ckpt_*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)

    for p in paths:
        try:
            torch.load(p, map_location="meta", weights_only=False)
            return p
        except Exception as e:
            print(f"skipping corrupt checkpoint {p.name}: {e}")

    return None


def _prune_old(ckpt_dir: Path, keep_last: int) -> None:
    """Keep only the N most recent checkpoints — they're ~4 GB each."""
    if keep_last <= 0:
        return
    paths = sorted(ckpt_dir.glob("ckpt_*.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[keep_last:]:
        p.unlink(missing_ok=True)


def push_to_hub(path: Path, repo_id: str, token: str | None = None) -> None:
    """Push a checkpoint to the HF Hub.

    Storage on a stopped pod still bills; the Hub is free and survives
    termination. Worth doing periodically for long runs.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=Path(path).name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"pushed {Path(path).name} -> {repo_id}")