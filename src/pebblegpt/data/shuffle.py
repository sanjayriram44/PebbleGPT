"""Combine per-source shards at the target mixture and shuffle at the
SEQUENCE level.

Sequence-level shuffling is not optional: the Playbook traced a noisy loss
curve to a dataloader that read sequences sequentially from each document,
letting a single long low-quality file fill an entire batch.

Memory note: this builds an index of (shard, record_offset) pairs and streams
records out one output shard at a time, so peak RAM is one output shard
(~200 MB) rather than the whole dataset.
"""

from pathlib import Path

import numpy as np

from pebblegpt.data.tokenize import RECORD_LEN, DTYPE

OUT_SHARD_RECORDS = 50_000   # ~200 MB per shard as uint16
ITEMSIZE = np.dtype(DTYPE).itemsize


def _build_index(in_dir: Path, sources: list[str]) -> list[tuple[Path, int]]:
    """Index every record as (shard_path, record_index) without reading data.

    Record counts come from file size, so nothing is paged into memory here.
    """
    index: list[tuple[Path, int]] = []

    for name in sources:
        paths = sorted(in_dir.glob(f"{name}_*.bin"))
        if not paths:
            raise FileNotFoundError(f"no shards for '{name}' in {in_dir}")

        n_source = 0
        for p in paths:
            n_rec = p.stat().st_size // (RECORD_LEN * ITEMSIZE)
            index.extend((p, i) for i in range(n_rec))
            n_source += n_rec

        print(f"{name}: {n_source:,} records ({n_source * RECORD_LEN:,} tokens)")

    return index


def combine_and_shuffle(in_dir: Path = Path("data/tokenized"),
                        out_dir: Path = Path("data/train"),
                        sources: list[str] | None = None,
                        seed: int = 1337) -> int:
    """Pool all sources' records, shuffle, and write final training shards.

    Mixture ratios are already baked in by tokenize.py's per-source token
    targets, so this only randomizes order.
    """
    from pebblegpt.data.download import MIXTURE

    sources = sources or list(MIXTURE.keys())
    in_dir, out_dir = Path(in_dir), Path(out_dir)

    if in_dir.resolve() == out_dir.resolve():
        raise ValueError("in_dir and out_dir must differ — writing into the "
                         "directory being read would corrupt the shards")

    out_dir.mkdir(parents=True, exist_ok=True)

    index = _build_index(in_dir, sources)
    n = len(index)
    print(f"total: {n:,} records ({n * RECORD_LEN:,} tokens)")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    open_maps: dict[Path, np.memmap] = {}

    def get_map(path: Path) -> np.memmap:
        if path not in open_maps:
            open_maps[path] = np.memmap(path, dtype=DTYPE, mode="r")
        return open_maps[path]

    n_written = 0
    for shard_idx, start in enumerate(range(0, n, OUT_SHARD_RECORDS)):
        chunk = order[start:start + OUT_SHARD_RECORDS]
        out = np.empty((len(chunk), RECORD_LEN), dtype=DTYPE)

        for j, k in enumerate(chunk):
            path, rec_i = index[k]
            m = get_map(path)
            s = rec_i * RECORD_LEN
            out[j] = m[s:s + RECORD_LEN]

        path = out_dir / f"train_{shard_idx:04d}.bin"
        out.tofile(path)
        n_written += out.shape[0]
        print(f"  wrote {path.name}  ({out.shape[0]:,} records)")

    open_maps.clear()
    return n_written


def reshuffle_for_epoch(in_dir: Path = Path("data/train"),
                        epoch: int = 1,
                        base_seed: int = 1337) -> int:
    """Reshuffle existing training shards into a fresh directory.

    The Playbook generated shuffled sequences per epoch with different seeds
    to avoid repeating shuffle patterns across epochs. Only needed if you
    train for more than one epoch — at Chinchilla you likely won't.
    """
    return combine_and_shuffle(
        in_dir=in_dir,
        out_dir=Path(f"data/train_epoch{epoch}"),
        sources=["train"],
        seed=base_seed + epoch,
    )