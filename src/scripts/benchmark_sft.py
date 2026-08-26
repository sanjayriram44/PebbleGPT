"""Sweep batch size / sequence length to find the best config for a given GPU.

    uv run python scripts/benchmark_sft.py
"""

import time
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def bench(model, tok, device, batch_size, seq_len, steps=5, use_amp=True):
    x = torch.randint(0, tok.vocab_size, (batch_size, seq_len), device=device)
    labels = x.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # warmup — first CUDA call always has overhead
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        out = model(input_ids=x, labels=labels)
        out.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(input_ids=x, labels=labels)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    tok_s = (batch_size * seq_len * steps) / dt
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_mem = float("nan")
    return tok_s, peak_mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="strectelite/PebbleGPT-320M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024])
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"vram: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    ).to(device)
    model.train()

    print(f"\n{'batch':>6} {'seq_len':>8} {'tok/s':>10} {'peak_mem_GB':>12}")
    for seq_len in args.seq_lens:
        for bs in args.batch_sizes:
            try:
                tok_s, mem = bench(model, tok, device, bs, seq_len)
                print(f"{bs:>6} {seq_len:>8} {tok_s:>10,.0f} {mem:>12.2f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{bs:>6} {seq_len:>8} {'OOM':>10} {'-':>12}")
                torch.cuda.empty_cache()
                break


if __name__ == "__main__":
    main()