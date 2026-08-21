"""Tokenize, pack into fixed-length sequences, and write per-source shards.

Each stored record is SEQ_LEN + 1 tokens: the loader slices it into
x = record[:-1] and y = record[1:] for next-token prediction.

Documents are tokenized in batches — HF's fast tokenizers are Rust-backed and
parallelize across cores on batched input, but run effectively single-threaded
when called one document at a time.
"""

import os
import time
from pathlib import Path

import numpy as np
import transformers
from transformers import AutoTokenizer

from pebblegpt.data.download import load_source, token_budget
from pebblegpt.data.filter import keep_document

# Must be set before the tokenizer is constructed.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

# Documents routinely exceed the tokenizer's declared model_max_length; we
# repack everything into SEQ_LEN chunks anyway, so the warning is noise.
transformers.logging.set_verbosity_error()

TOKENIZER = "HuggingFaceTB/SmolLM2-360M"
SEQ_LEN = 2048
RECORD_LEN = SEQ_LEN + 1        # +1 so each record yields aligned (x, y)
SHARD_TOKENS = 100_000_000      # ~200 MB per shard as uint16
DTYPE = np.uint16               # 49,152 vocab fits
TOKENIZE_BATCH = 1000           # documents per tokenizer call


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    assert tok.vocab_size <= np.iinfo(DTYPE).max, "vocab too large for uint16"
    return tok


def tokenize_source(name: str,
                    target_tokens: int,
                    out_dir: Path,
                    tokenizer=None,
                    streaming: bool = True,
                    log_every_s: float = 5.0,
                    batch_size: int = TOKENIZE_BATCH) -> int:
    """Stream one source, tokenize, pack, and write shards.

    Stops as soon as target_tokens is reached. Returns tokens actually written.
    """
    tokenizer = tokenizer or get_tokenizer()
    eos = tokenizer.eos_token_id
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = min(batch_size, max(1, target_tokens // 2000))
    ds, text_col = load_source(name, streaming=streaming)

    buffer: list[int] = []
    records: list[np.ndarray] = []
    pending: list[str] = []
    written = 0
    shard_idx = 0
    n_docs = 0
    n_dropped = 0
    n_seen = 0

    t0 = time.time()
    last_log = t0

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

    def carve_records():
        """Move complete records out of the buffer, flushing shards as needed."""
        nonlocal buffer, written
        while len(buffer) >= RECORD_LEN:
            records.append(np.array(buffer[:RECORD_LEN], dtype=DTYPE))
            buffer = buffer[RECORD_LEN:]
            written += RECORD_LEN
            if written % SHARD_TOKENS < RECORD_LEN:
                flush_shard()

    def flush_pending():
        """Tokenize the accumulated batch and append to the buffer."""
        nonlocal pending
        if not pending:
            return
        encoded = tokenizer(pending, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            buffer.extend(ids)
            buffer.append(eos)
            carve_records()
            if written >= target_tokens:
                break
        pending = []

    def log_progress(force: bool = False):
        nonlocal last_log
        now = time.time()
        if not force and now - last_log < log_every_s:
            return
        last_log = now
        elapsed = now - t0
        pct = 100 * written / target_tokens if target_tokens else 0
        rate = written / elapsed if elapsed > 0 else 0
        eta = (target_tokens - written) / rate if rate > 0 else float("inf")
        print(f"  {name}: {written:,}/{target_tokens:,} ({pct:5.1f}%)  "
              f"seen={n_seen:,} kept={n_docs:,} dropped={n_dropped:,}  "
              f"{rate:,.0f} tok/s  eta {eta:,.0f}s", flush=True)

    for doc in ds:
        n_seen += 1
        log_progress()          # fires even if every doc is being dropped

        text = doc.get(text_col)
        if not keep_document(text):
            n_dropped += 1
            continue

        pending.append(text)
        n_docs += 1

        if len(pending) >= batch_size:
            flush_pending()
            if written >= target_tokens:
                break

    # tokenize whatever is left in the final partial batch
    if written < target_tokens:
        flush_pending()

    flush_shard()
    log_progress(force=True)
    print(f"{name}: {written:,} tokens from {n_docs:,} docs "
          f"({n_dropped:,}/{n_seen:,} dropped)", flush=True)

    if written < target_tokens * 0.95:
        print(f"  WARNING: {name} produced {written:,} of {target_tokens:,} "
              f"({100 * written / target_tokens:.0f}%) — source exhausted or "
              f"over-filtered", flush=True)

    return written


def tokenize_all(total_tokens: int,
                 out_dir: Path = Path("data/tokenized"),
                 sources: list[str] | None = None,
                 streaming: bool = True) -> dict[str, int]:
    """Tokenize each source to its share of the total token budget."""
    tokenizer = get_tokenizer()
    budget = token_budget(total_tokens)
    sources = sources or list(budget.keys())

    results = {}
    for name in sources:
        target = budget[name]
        print(f"\n=== {name}: target {target:,} tokens ===", flush=True)
        results[name] = tokenize_source(
            name, target, out_dir,
            tokenizer=tokenizer, streaming=streaming,
        )
    return results