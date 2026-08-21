"""Export a checkpoint and evaluate it.

    uv run python scripts/evaluate.py --ckpt checkpoints/ckpt_final.pt
    uv run python scripts/evaluate.py --ckpt ... --tasks hellaswag --device mps
"""

import argparse
from pathlib import Path

from pebblegpt.eval.harness import export_hf_model, run_eval, CORE_TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample N questions per task for speed")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or Path("exports") / args.ckpt.stem
    export_hf_model(args.ckpt, out_dir)

    if not args.export_only:
        run_eval(
            out_dir,
            tasks=args.tasks or CORE_TASKS,
            device=args.device,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()