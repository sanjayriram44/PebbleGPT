"""Break down why documents are being dropped, per source."""

import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import argparse

from pebblegpt.data.download import MIXTURE, load_source
from pebblegpt.data.filter import is_too_short, has_excessive_repetition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(MIXTURE.keys()))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    for name in args.sources:
        ds, col = load_source(name, streaming=True)
        short = rep = ok = 0
        lengths = []

        for i, doc in enumerate(ds):
            if i >= args.n:
                break
            t = doc.get(col) or ""
            lengths.append(len(t))
            if not t or is_too_short(t):
                short += 1
            elif has_excessive_repetition(t):
                rep += 1
            else:
                ok += 1

        total = short + rep + ok
        lengths.sort()
        median = lengths[len(lengths) // 2] if lengths else 0
        print(f"{name}: kept={ok}/{total} ({100*ok/total:.0f}%)  "
              f"short={short}  repetition={rep}  median_chars={median:,}")


if __name__ == "__main__":
    main()