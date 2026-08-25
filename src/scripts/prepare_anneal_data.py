"""Prepare the annealing dataset: baseline replay + PIQA/ARC-targeted data.

    uv run python src/scripts/prepare_anneal_data.py --total-tokens 300_000_000
"""

import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import argparse
from pathlib import Path

import numpy as np

from pebblegpt.data.anneal_sources import anneal_token_budget, ANNEAL_SOURCES, load_anneal_source
from pebblegpt.data.filter import keep_document
from pebblegpt.data.tokenize import get_tokenizer, RECORD_LEN, DTYPE, SHARD_TOKENS
from pebblegpt.data.shuffle import combine_and_shuffle


def tokenize_anneal_source(name: str, target_tokens: int, out_dir: Path, tokenizer) -> int:
    """Same packing logic as pebblegpt.data.tokenize.tokenize_source, but reads
    from ANNEAL_SOURCES instead of the pretraining SOURCES registry."""
    eos = tokenizer.eos_token_id
    out_dir.mkdir(parents=True, exist_ok=True)
    ds, text_col = load_anneal_source(name, streaming=True)

    buffer: list[int] = []
    records: list[np.ndarray] = []
    written = 0
    shard_idx = 0
    n_seen = 0
    n_dropped = 0

    def flush_shard():
        nonlocal records, shard_idx
        if not records:
            return
        arr = np.concatenate(records).astype(DTYPE)
        path = out_dir / f"{name}_{shard_idx:04d}.bin"
        arr.tofile(path)
        print(f"  wrote {path.name}  ({arr.size:,} tokens)", flush=True)
        records = []
        shard_idx += 1

    for doc in ds:
        n_seen += 1
        text = doc.get(text_col)
        if not keep_document(text):
            n_dropped += 1
            continue

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buffer.extend(ids)
        buffer.append(eos)

        while len(buffer) >= RECORD_LEN:
            records.append(np.array(buffer[:RECORD_LEN], dtype=DTYPE))
            buffer = buffer[RECORD_LEN:]
            written += RECORD_LEN
            if written % SHARD_TOKENS < RECORD_LEN:
                flush_shard()

        if written >= target_tokens:
            break

    flush_shard()
    print(f"{name}: {written:,} tokens ({n_dropped:,}/{n_seen:,} dropped)", flush=True)

    if written < target_tokens * 0.95:
        print(f"  WARNING: {name} produced {written:,} of {target_tokens:,} "
              f"({100 * written / target_tokens:.0f}%) — source exhausted or "
              f"over-filtered", flush=True)

    return written


def sample_baseline_replay(tokenized_dir: Path, out_dir: Path,
                           target_tokens: int, seed: int = 1337) -> int:
    """Sample existing pretraining shards (all 4 sources, already at their
    original ratios) for the baseline-replay portion of the anneal mix."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for path in sorted(tokenized_dir.glob("*.bin")):
        n_rec = path.stat().st_size // (RECORD_LEN * np.dtype(DTYPE).itemsize)
        index.extend((path, i) for i in range(n_rec))

    if not index:
        raise FileNotFoundError(f"no shards found in {tokenized_dir}")

    n_needed = target_tokens // RECORD_LEN
    if n_needed > len(index):
        print(f"WARNING: only {len(index):,} baseline records available, "
              f"wanted {n_needed:,}")
        n_needed = len(index)

    chosen = rng.choice(len(index), size=n_needed, replace=False)

    open_maps: dict[Path, np.memmap] = {}

    def get_map(p: Path) -> np.memmap:
        if p not in open_maps:
            open_maps[p] = np.memmap(p, dtype=DTYPE, mode="r")
        return open_maps[p]

    out = np.empty((n_needed, RECORD_LEN), dtype=DTYPE)
    for j, k in enumerate(chosen):
        path, rec_i = index[k]
        m = get_map(path)
        s = rec_i * RECORD_LEN
        out[j] = m[s:s + RECORD_LEN]

    path = out_dir / "baseline_replay_0000.bin"
    out.tofile(path)
    written = out.size
    print(f"  wrote {path.name}  ({written:,} tokens, {n_needed:,} records)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-tokens", type=int, default=300_000_000)
    ap.add_argument("--tokenized-dir", type=Path, default=Path("data/tokenized"))
    ap.add_argument("--anneal-tokenized-dir", type=Path, default=Path("data/anneal_tokenized"))
    ap.add_argument("--anneal-train-dir", type=Path, default=Path("data/anneal_train"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-shuffle", action="store_true")
    args = ap.parse_args()

    budget = anneal_token_budget(args.total_tokens)
    baseline_target = int(args.total_tokens * 0.40)

    print(f"Anneal target: {args.total_tokens:,} tokens")
    print(f"  baseline_replay: {baseline_target:,}")
    for name, tok in budget.items():
        print(f"  {name}: {tok:,}")

    if not args.skip_download:
        tokenizer = get_tokenizer()
        for name, target in budget.items():
            print(f"\n=== {name}: target {target:,} tokens ===", flush=True)
            tokenize_anneal_source(name, target, args.anneal_tokenized_dir, tokenizer)

    print("\n=== sampling baseline replay ===")
    sample_baseline_replay(args.tokenized_dir, args.anneal_tokenized_dir,
                           baseline_target, seed=args.seed)

    if not args.skip_shuffle:
        print("\n=== shuffling anneal set ===")
        combine_and_shuffle(
            in_dir=args.anneal_tokenized_dir,
            out_dir=args.anneal_train_dir,
            sources=["baseline_replay"] + list(ANNEAL_SOURCES.keys()),
            seed=args.seed,
        )
        print(f"\nDone: anneal data ready in {args.anneal_train_dir}")


if __name__ == "__main__":
    main()