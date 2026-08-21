"""Training loop: gradient accumulation, BF16 autocast, clipping, MFU.

Deliberately plain — no Trainer abstraction. Everything the run depends on
is visible here.
"""

import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import torch

from pebblegpt.data.loader import build_dataloader
from pebblegpt.model.model import PebbleGPT
from pebblegpt.train.optimizer import build_optimizer
from pebblegpt.train.scheduler import WSDScheduler
from pebblegpt.utils.checkpoint import save_checkpoint, load_checkpoint, find_latest
from pebblegpt.utils.logging import Logger, compute_mfu, guess_peak_flops


@dataclass
class TrainConfig:
    # data
    data_dir: Path = Path("data/train")
    num_workers: int = 4

    # batch — global = micro_batch * grad_accum * seq_len
    micro_batch_size: int = 8
    grad_accum_steps: int = 32
    seq_len: int = 2048

    # schedule
    total_tokens: int = 6_400_000_000
    peak_lr: float = 5e-4
    min_lr: float = 5e-5
    warmup_steps: int = 2000
    decay_fraction: float = 0.10

    # optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # runtime
    compile: bool = True
    dtype: str = "bfloat16"
    device: str = "cuda"
    seed: int = 1337

    # checkpointing / logging
    ckpt_dir: Path = Path("checkpoints")
    ckpt_every: int = 500
    keep_last: int = 3
    log_every: int = 10
    run_name: str = "pebblegpt-320m"
    use_wandb: bool = False
    wandb_entity: str | None = None

    # model
    model_kwargs: dict = field(default_factory=dict)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    @property
    def total_steps(self) -> int:
        return self.total_tokens // self.tokens_per_step


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        fallback = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"cuda unavailable, falling back to {fallback}")
        return torch.device(fallback)
    return torch.device(requested)


def train(cfg: TrainConfig, resume: bool = True) -> None:
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    device_type = device.type

    # bfloat16 needs no GradScaler, unlike fp16
    autocast_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    use_autocast = device_type in ("cuda", "cpu")

    model = PebbleGPT(**cfg.model_kwargs).to(device)
    n_params = model.num_params()
    print(f"model: {n_params/1e6:.1f}M params")
    print(f"batch: {cfg.tokens_per_step:,} tokens/step  "
          f"({cfg.micro_batch_size} x {cfg.grad_accum_steps} x {cfg.seq_len})")
    print(f"schedule: {cfg.total_steps:,} steps for {cfg.total_tokens:,} tokens")

    optimizer = build_optimizer(
        model, lr=cfg.peak_lr, weight_decay=cfg.weight_decay,
        beta1=cfg.beta1, beta2=cfg.beta2, device_type=device_type,
    )
    scheduler = WSDScheduler(
        optimizer, total_steps=cfg.total_steps, peak_lr=cfg.peak_lr,
        min_lr=cfg.min_lr, warmup_steps=cfg.warmup_steps,
        decay_fraction=cfg.decay_fraction,
    )

    start_step, tokens_seen = 0, 0
    if resume:
        latest = find_latest(cfg.ckpt_dir)
        if latest is not None:
            meta = load_checkpoint(latest, model, optimizer, scheduler,
                                   device=str(device))
            start_step = meta["step"]
            tokens_seen = meta["tokens_seen"]
            print(f"resumed from {latest.name} at step {start_step:,}")

    # compile after loading so state dict keys aren't prefixed with _orig_mod
    if cfg.compile and device_type == "cuda":
        print("compiling model...")
        model = torch.compile(model)

    loader = build_dataloader(
        data_dir=cfg.data_dir,
        batch_size=cfg.micro_batch_size,
        num_workers=cfg.num_workers,
    )

    logger = Logger(
        run_name=cfg.run_name,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        config={k: str(v) for k, v in asdict(cfg).items()},
        log_every=cfg.log_every,
    )
    peak_flops = guess_peak_flops()

    def batches():
        """Infinite stream — loops the loader when exhausted."""
        while True:
            for x, y in loader:
                yield x, y

    stream = batches()
    model.train()
    t0 = time.time()
    step = start_step

    try:
        for step in range(start_step, cfg.total_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0

            for _ in range(cfg.grad_accum_steps):
                x, y = next(stream)
                if cfg.seq_len < x.size(1):
                    x = x[:, :cfg.seq_len]
                    y = y[:, :cfg.seq_len]
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if use_autocast:
                    with torch.autocast(device_type=device_type,
                                        dtype=autocast_dtype):
                        _, loss = model(x, targets=y)
                else:
                    _, loss = model(x, targets=y)

                # scale so accumulated grads equal the mean over the full batch
                loss = loss / cfg.grad_accum_steps
                loss.backward()
                loss_total += loss.item()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            optimizer.step()
            lr = scheduler.step()

            tokens_seen += cfg.tokens_per_step

            if step % cfg.log_every == 0:
                if device_type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                t0 = time.time()
                tok_s = cfg.tokens_per_step * cfg.log_every / dt if dt > 0 else 0
                logger.log(step, {
                    "loss": loss_total,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tok_per_s": tok_s,
                    "mfu": compute_mfu(tok_s, n_params, peak_flops),
                    "tokens_seen": tokens_seen,
                })

            if step > 0 and step % cfg.ckpt_every == 0:
                raw = getattr(model, "_orig_mod", model)   # unwrap torch.compile
                path = cfg.ckpt_dir / f"ckpt_{step:07d}.pt"
                save_checkpoint(path, raw, optimizer, scheduler,
                                step=step, tokens_seen=tokens_seen,
                                config=asdict(cfg), keep_last=cfg.keep_last)
                print(f"  saved {path.name}", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted — saving before exit")
        raw = getattr(model, "_orig_mod", model)
        save_checkpoint(cfg.ckpt_dir / f"ckpt_{step:07d}.pt", raw, optimizer,
                        scheduler, step=step, tokens_seen=tokens_seen,
                        config=asdict(cfg), keep_last=cfg.keep_last)
        raise

    else:
        # only on clean completion — a crash must not leave a "final" marker
        raw = getattr(model, "_orig_mod", model)
        save_checkpoint(cfg.ckpt_dir / "ckpt_final.pt", raw, optimizer,
                        scheduler, step=cfg.total_steps, tokens_seen=tokens_seen,
                        config=asdict(cfg), keep_last=cfg.keep_last + 1)

    finally:
        logger.close()
        print(f"done: {tokens_seen:,} tokens")