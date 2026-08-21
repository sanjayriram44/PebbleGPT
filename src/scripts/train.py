"""Launch a training run.

    uv run python scripts/train.py --smoke --device mps
    uv run python scripts/train.py --wandb
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
    ap.add_argument("--total-tokens", type=int, default=6_400_000_000)
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="Strectelite")
    ap.add_argument("--run-name", default="pebblegpt-320m")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-limit", type=int, default=1000)
    args = ap.parse_args()

    if args.smoke:
        cfg = TrainConfig(
            total_tokens=200 * 1 * 2 * 512,
            micro_batch_size=1,
            grad_accum_steps=2,
            seq_len=512,
            warmup_steps=20,
            ckpt_every=50,
            log_every=5,
            compile=False,
            num_workers=0,
            device=args.device,
            ckpt_dir=Path("checkpoints/smoke"),
            run_name="smoke",
            use_wandb=args.wandb,
            wandb_entity=args.wandb_entity,
            model_kwargs={"max_seq_len": 512},
            eval_every=args.eval_every if args.eval_every else 0,
            eval_limit=args.eval_limit,
            eval_tasks=["hellaswag"],
        )
    else:
        cfg = TrainConfig(
            total_tokens=args.total_tokens,
            micro_batch_size=args.micro_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            device=args.device,
            compile=not args.no_compile,
            use_wandb=args.wandb,
            wandb_entity=args.wandb_entity,
            run_name=args.run_name,
            eval_every=args.eval_every,
        )

    print(f"steps: {cfg.total_steps:,}  tokens/step: {cfg.tokens_per_step:,}")
    train(cfg, resume=not args.no_resume)


if __name__ == "__main__":
    main()