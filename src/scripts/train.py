"""Launch a training run.

    uv run python scripts/train.py --smoke --device mps
    uv run python scripts/train.py --wandb

    # annealing (warm restart from a fully-decayed checkpoint):
    uv run python scripts/train.py \
        --data-dir data/anneal_train --ckpt-dir checkpoints/anneal \
        --anneal-from checkpoints/ckpt_final.pt --anneal-peak-lr 1.2e-4 \
        --total-tokens 300_000_000 --decay-fraction 0.85 \
        --wandb --run-name pebblegpt-anneal --no-resume
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pebblegpt.train.loop import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="short run against local test data")
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    ap.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--total-tokens", type=int, default=6_400_000_000)
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=32)
    ap.add_argument("--warmup-steps", type=int, default=2000)
    ap.add_argument("--decay-fraction", type=float, default=0.10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="Strectelite")
    ap.add_argument("--run-name", default="pebblegpt-320m")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-limit", type=int, default=1000)
    ap.add_argument("--eval-tasks", nargs="+", default=None)

    # annealing
    ap.add_argument("--anneal-from", type=Path, default=None)
    ap.add_argument("--anneal-peak-lr", type=float, default=1.2e-4)

    args = ap.parse_args()

    if args.smoke:
        cfg = TrainConfig(
            data_dir=args.data_dir,
            total_tokens=200 * 1 * 2 * 512,
            micro_batch_size=1,
            grad_accum_steps=2,
            seq_len=512,
            warmup_steps=20,
            ckpt_dir=Path("checkpoints/smoke"),
            ckpt_every=50,
            log_every=5,
            compile=False,
            num_workers=0,
            device=args.device,
            run_name="smoke",
            use_wandb=args.wandb,
            wandb_entity=args.wandb_entity,
            eval_every=args.eval_every if args.eval_every else 0,
            eval_limit=args.eval_limit,
            eval_tasks=args.eval_tasks or ["hellaswag"],
            model_kwargs={"max_seq_len": 512},
        )
    else:
        cfg = TrainConfig(
            data_dir=args.data_dir,
            ckpt_dir=args.ckpt_dir,
            total_tokens=args.total_tokens,
            micro_batch_size=args.micro_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            warmup_steps=args.warmup_steps,
            decay_fraction=args.decay_fraction,
            device=args.device,
            compile=not args.no_compile,
            use_wandb=args.wandb,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
            eval_every=args.eval_every,
            eval_limit=args.eval_limit,
            eval_tasks=args.eval_tasks or ["hellaswag", "piqa", "arc_easy"],
            anneal_from=args.anneal_from,
            anneal_peak_lr=args.anneal_peak_lr,
        )

    print(f"steps: {cfg.total_steps:,}  tokens/step: {cfg.tokens_per_step:,}")
    train(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()