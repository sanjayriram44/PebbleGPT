"""Memory-mapped dataset over the final shuffled shards."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pebblegpt.data.tokenize import RECORD_LEN, DTYPE, SEQ_LEN


class PackedDataset(Dataset):
    """Serves (x, y) pairs of length SEQ_LEN from packed token shards.

    Records are stored as SEQ_LEN+1 tokens so x and y are aligned without
    dropping a token: x = record[:-1], y = record[1:].
    """

    def __init__(self, data_dir: Path = Path("data/train"), pattern: str = "train_*.bin"):
        self.paths = sorted(Path(data_dir).glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"no shards matching {pattern} in {data_dir}")

        self.shards = []
        self.offsets = [0]
        for p in self.paths:
            a = np.memmap(p, dtype=DTYPE, mode="r")
            n = a.size // RECORD_LEN
            self.shards.append((p, n))
            self.offsets.append(self.offsets[-1] + n)

        self.n_records = self.offsets[-1]
        self._cache: dict[int, np.memmap] = {}

    def __len__(self) -> int:
        return self.n_records

    def _get_shard(self, shard_idx: int) -> np.memmap:
        # memmaps are opened lazily so DataLoader workers don't share handles
        if shard_idx not in self._cache:
            path, _ = self.shards[shard_idx]
            self._cache[shard_idx] = np.memmap(path, dtype=DTYPE, mode="r")
        return self._cache[shard_idx]

    def __getitem__(self, idx: int):
        shard_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        local = idx - self.offsets[shard_idx]

        shard = self._get_shard(shard_idx)
        start = local * RECORD_LEN
        record = np.asarray(shard[start:start + RECORD_LEN]).astype(np.int64)

        x = torch.from_numpy(record[:-1])
        y = torch.from_numpy(record[1:])
        return x, y

    @property
    def n_tokens(self) -> int:
        return self.n_records * SEQ_LEN


def build_dataloader(data_dir: Path = Path("data/train"),
                     batch_size: int = 8,
                     num_workers: int = 4,
                     shuffle: bool = False) -> DataLoader:
    """Shards are already shuffled offline, so shuffle=False by default."""
    ds = PackedDataset(data_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )