"""Run the full data pipeline: tokenize -> shuffle.

Usage:
    uv run python scripts/prepare_data.py --total-tokens 1_000_000
    uv run python scripts/prepare_data.py --sources fineweb-edu finemath
    caffeinate -i uv run python scripts/prepare_data.py   # full budget, macOS
"""

import os

# Must be set before datasets/huggingface_hub are imported.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import argparse
from pathlib import Path

from pebblegpt.data.download import CHINCHILLA_TOKENS, MIXTURE, token_budget
from pebblegpt.data.tokenize import tokenize_all
from pebblegpt.data.shuffle import combine_and_shuffle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-tokens", type=int, default=CHINCHILLA_TOKENS)
    ap.add_argument("--sources", nargs="+", default=None,
                    choices=list(MIXTURE.keys()),
                    help="subset of sources to process (default: all)")
    ap.add_argument("--tokenized-dir", type=Path, default=Path("data/tokenized"))
    ap.add_argument("--train-dir", type=Path, default=Path("data/train"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-streaming", action="store_true",
                    help="download shards instead of streaming (better for "
                         "large token budgets and flaky connections)")
    ap.add_argument("--skip-tokenize", action="store_true")
    ap.add_argument("--skip-shuffle", action="store_true")
    args = ap.parse_args()

    sources = args.sources or list(MIXTURE.keys())
    budget = token_budget(args.total_tokens)

    print(f"Target: {args.total_tokens:,} tokens")
    for name in sources:
        print(f"  {name}: {budget[name]:,}")
    if args.sources:
        skipped = [s for s in MIXTURE if s not in sources]
        if skipped:
            print(f"  (skipping: {', '.join(skipped)})")

    if not args.skip_tokenize:
        tokenize_all(
            args.total_tokens,
            out_dir=args.tokenized_dir,
            sources=sources,
            streaming=not args.no_streaming,
        )

    if not args.skip_shuffle:
        print("\n=== shuffling ===")
        n = combine_and_shuffle(
            in_dir=args.tokenized_dir,
            out_dir=args.train_dir,
            sources=sources,
            seed=args.seed,
        )
        print(f"\nDone: {n:,} records ready in {args.train_dir}")


if __name__ == "__main__":
    main()