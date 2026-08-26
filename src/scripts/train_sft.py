"""SFT training run.

    uv run python scripts/train_sft.py \
        --batch-size 8 --seq-len 512 --n-convos 30000 --epochs 2 \
        --push-to-hub strectelite/PebbleGPT-320M-Instruct
"""

import argparse
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from pebblegpt.data.sft_dataset import SFTDataset, load_smoltalk


@torch.no_grad()
def sample(model, tok, device, prompt, max_new=80):
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    model.eval()
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=True,
        temperature=0.7, top_k=40,
        eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
    )
    model.train()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="strectelite/PebbleGPT-320M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-convos", type=int, default=30_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/pebblegpt-sft"))
    ap.add_argument("--push-to-hub", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("loading conversations...")
    convos = load_smoltalk(n_examples=args.n_convos)
    print(f"got {len(convos)} conversations")

    ds = SFTDataset(convos, tok, max_len=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    warmup_steps = max(10, int(total_steps * 0.03))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps, anneal_strategy="cos",
    )
    print(f"total_steps={total_steps}  warmup_steps={warmup_steps}")

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            x = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model(input_ids=x, labels=labels)
                loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps}  "
                      f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)
            step += 1

    print("\nSFT complete — sampling a few outputs:")
    for q in ["What is the capital of France?",
              "Write a short poem about the ocean.",
              "Explain what a for loop does."]:
        print(f"\nQ: {q}")
        print(f"A: {sample(model, tok, device, q)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)

    model_pkg = Path(__file__).parent.parent / "pebblegpt" / "model"
    for name in ("configuration.py", "modeling.py", "block.py", "attention.py", "SwiGLU.py"):
        src = (model_pkg / name).read_text().replace("from pebblegpt.model.", "from .")
        (args.out_dir / name).write_text(src)

    print(f"\nsaved to {args.out_dir}")

    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(args.out_dir), repo_id=args.push_to_hub, repo_type="model")
        print(f"pushed to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()