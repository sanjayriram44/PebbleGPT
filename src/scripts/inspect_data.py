"""Decode packed records back to text and eyeball them."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from pebblegpt.data.loader import PackedDataset
from pebblegpt.data.tokenize import TOKENIZER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--chars", type=int, default=1200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ds = PackedDataset(args.data_dir)

    print(f"{len(ds):,} records, {ds.n_tokens:,} tokens\n")

    for i in range(min(args.n, len(ds))):
        x, y = ds[i]
        assert x.shape == (2048,) and y.shape == (2048,)
        assert torch.equal(x[1:], y[:-1]), "x/y shift is wrong"

        text = tok.decode(x.tolist())
        n_eos = (x == tok.eos_token_id).sum().item()

        print(f"--- record {i}  ({n_eos} EOS markers = ~{n_eos + 1} docs) ---")
        print(text[:args.chars])
        print("...\n")


if __name__ == "__main__":
    main()