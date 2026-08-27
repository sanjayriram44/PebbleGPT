# PebbleGPT

An exploratory study in training a language model from scratch on a small budget.

The question: how far can one person get on roughly $60 of rented compute, writing
the architecture by hand and curating the training data themselves?

This repository contains the full pipeline: architecture, data curation,
pretraining, annealing, supervised fine-tuning, and evaluation. Every component was
written from scratch rather than adapted from an existing training framework.

Models: PebbleGPT-320M (base), PebbleGPT-320M-Instruct (chat)

## Contents

1. Quick start
2. Results
3. Model
4. Training
5. Repository structure
6. Reproducing
7. Compute cost
8. What went wrong
9. Acknowledgements
10. License

## Quick start

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "strectelite/PebbleGPT-320M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).cuda()

    prompt = "Explain what recursion is in simple terms."
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_k=40,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

## Results

Zero-shot, full test sets, evaluated with lm-evaluation-harness.

| Benchmark | After pretraining | After annealing | After SFT |
|---|---|---|---|
| HellaSwag (acc_norm) | 30.5 | 40.4 | 40.6 |
| PIQA (acc_norm) | 55.5 | 67.7 | 66.8 |
| ARC-easy (acc_norm) | 37.7 | 56.4 | 53.8 |
| ARC-challenge (acc_norm) | n/a | 31.1 | 30.0 |
| Training loss | 2.507 | 2.067 | n/a |

Instruction following, measured on IFEval after supervised fine-tuning:

| Metric | Score |
|---|---|
| inst-level loose | 27.8 |
| inst-level strict | 27.2 |
| prompt-level loose | 14.4 |
| prompt-level strict | 13.7 |

The annealing phase cost about $1 and took 30 minutes. It produced roughly ten to
nineteen points of improvement across the three benchmarks it targeted, the single
highest return step in the project.

## Model

Dense decoder-only transformer.

| Parameter | Value |
|---|---|
| Total parameters | 320.9M |
| Non-embedding parameters | 270.6M |
| Layers | 24 |
| Hidden size | 1024 |
| Intermediate size | 2816 |
| Attention | Grouped query attention |
| Attention heads | 16 query, 4 KV |
| Head dimension | 64 |
| Activation | SwiGLU |
| Normalization | RMSNorm, pre-norm |
| Positional encoding | RoPE, theta 10000 |
| Embeddings | Tied input and output |
| Context length | 2048 |
| Vocabulary | 49,152, SmolLM2 tokenizer |

### Design notes

Intermediate size of 2816, not 4096. SwiGLU uses three weight matrices where a
standard feedforward layer uses two. Using 8/3 of the hidden size instead of 4x keeps
the parameter count comparable.

Vocabulary of 49,152, not 128k. At 320M parameters a 128k vocabulary would put 37
percent of the budget into a lookup table. The smaller vocabulary reduces that to 14
percent, freeing roughly 80M parameters for transformer layers.

Grouped query attention at ratio 4. Sixteen query heads share four KV heads, cutting
the KV cache to a quarter of full multi-head attention with no measured quality loss
at this scale.

Warmup-Stable-Decay learning rate schedule. Unlike cosine decay, WSD does not require
committing to a total token count before training starts. The learning rate holds at
peak for most of the run and decays only in the final stretch.

## Training

Three phases, 10.4B tokens total.

### Phase 1: Pretraining

10B tokens, 19,073 steps at 524,288 tokens per step.

| Source | Share |
|---|---|
| FineWeb-Edu | 42.5% |
| DCLM-Edu, filtered to edu_int_score >= 3 | 42.5% |
| Python-Edu | 10% |
| FineMath-4+ | 5% |

AdamW, betas 0.9 and 0.95, weight decay 0.1 excluding embeddings and norms, gradient
clipping 1.0. WSD schedule, peak learning rate 5e-4, minimum 5e-5, 2,000 warmup
steps, 10 percent decay window.

Data pipeline: n-gram repetition filtering, tokenization, packing into fixed 2,049
token records with EOS separators, sequence-level shuffling, written as uint16
binary shards read through memory mapping.

One H100 SXM, 18 hours, 35 percent model FLOPs utilization. Final loss 2.507.

### Phase 2: Annealing

300M tokens of targeted continued pretraining, aimed at physical commonsense
reasoning and grade school science.

| Source | Share | Purpose |
|---|---|---|
| Baseline replay from phase 1 | 40% | Prevent narrowing |
| Cosmopedia wikihow | 20% | Procedural and physical reasoning |
| Cosmopedia stories | 10% | Everyday world knowledge |
| Cosmopedia openstax | 15% | Science curriculum |
| Cosmopedia khanacademy | 15% | Science curriculum |

Warm restart from the fully decayed phase 1 checkpoint, fresh optimizer state. Peak
learning rate 1.2e-4, roughly 25 percent of the original peak, 30 warmup steps, 85
percent decay window, 572 steps.

The 40 percent baseline replay exists to prevent catastrophic narrowing toward the
new domains. Cosmopedia is synthetic data generated by Mixtral-8x7B-Instruct-v0.1,
decontaminated by its authors against ARC, PIQA, HellaSwag, OpenBookQA, WinoGrande,
MMLU, and BoolQ using 10-gram overlap detection.

One RTX PRO 6000, 30 minutes. Final loss 2.067.

### Phase 3: Supervised fine-tuning

50,000 conversations from SmolTalk, 2 epochs, ChatML format.

Loss is computed only on assistant response tokens. User turns, template tokens, and
padding are masked with -100. Getting this masking exactly right matters: a subtle
bug here produces a plausible looking but useless model, so it has dedicated unit
tests.

AdamW at 2e-5, weight decay 0.01, OneCycleLR cosine schedule, 3 percent warmup,
batch size 32, sequence length 1,024, 3,126 steps.

One A100 80GB, 20 minutes.

## Repository structure

    src/pebblegpt/
      model/
        attention.py       Grouped query attention with RoPE and KV cache
        SwiGLU.py           Gated feedforward layer
        block.py             Transformer block, pre-norm
        model.py              PebbleGPT, the training-time model
        modeling.py            HuggingFace-compatible wrapper with generation
        configuration.py        HuggingFace config class
      data/
        download.py              Source registry and mixture ratios
        filter.py                 N-gram repetition filter
        tokenize.py                Batched tokenization, packing, sharding
        shuffle.py                  Sequence-level shuffle across sources
        loader.py                    Memory-mapped dataset
        sft_dataset.py                 ChatML formatting and loss masking
        anneal_sources.py                Annealing phase data sources
      train/
        scheduler.py       Warmup-Stable-Decay
        optimizer.py         AdamW with weight decay exclusions
        loop.py                Training loop, gradient accumulation, MFU tracking
      eval/
        harness.py       Export to HF format, run lm-evaluation-harness
      utils/
        checkpoint.py       Atomic saves with full resume state
        logging.py             Console, JSONL, and Weights and Biases

    src/scripts/
      prepare_data.py            Pretraining data pipeline
      prepare_anneal_data.py       Annealing data pipeline
      train.py                       Pretraining and annealing
      train_sft.py                     Supervised fine-tuning
      benchmark_sft.py                   Batch size and sequence length sweep
      evaluate.py                          Export and evaluate a checkpoint
      inspect_data.py                        Decode packed records for review
      diagnose_filter.py                       Break down filter rejection reasons

## Reproducing

    git clone https://github.com/sanjayriram44/PebbleGPT.git
    cd PebbleGPT
    uv sync

    cat > .env <<'EOF'
    HF_TOKEN=hf_...
    WANDB_API_KEY=...
    WANDB_ENTITY=your_entity
    EOF

Data preparation. Run this on a CPU instance, not a GPU.

    uv run python src/scripts/prepare_data.py --total-tokens 10_000_000_000

Pretraining.

    uv run python src/scripts/train.py \
        --micro-batch-size 16 --grad-accum-steps 16 \
        --total-tokens 10_000_000_000 --eval-every 1000 --wandb

Annealing.

    uv run python src/scripts/prepare_anneal_data.py --total-tokens 300_000_000

    uv run python src/scripts/train.py \
        --data-dir data/anneal_train --ckpt-dir checkpoints/anneal \
        --anneal-from checkpoints/ckpt_final.pt --anneal-peak-lr 1.2e-4 \
        --total-tokens 300_000_000 --warmup-steps 30 --decay-fraction 0.85 \
        --micro-batch-size 16 --grad-accum-steps 16 --no-resume --wandb

Supervised fine-tuning. Fits comfortably in a free Colab session.

    uv run python src/scripts/benchmark_sft.py

    uv run python src/scripts/train_sft.py \
        --batch-size 32 --seq-len 1024 --n-convos 50000 --epochs 2 \
        --push-to-hub your-username/PebbleGPT-320M-Instruct

Evaluation.

    uv run python src/scripts/evaluate.py \
        --ckpt checkpoints/anneal/ckpt_final.pt \
        --tasks hellaswag piqa arc_easy arc_challenge --device cuda

## Compute cost

| Phase | Hardware | Duration | Cost |
|---|---|---|---|
| Data preparation | 32 vCPU instance at $1.28/hr | 5 hours | $6 |
| Pretraining | H100 SXM at $3.31/hr | 18 hours | $34 |
| Annealing | RTX PRO 6000 at $2.00/hr | 30 minutes | $1 |
| Supervised fine-tuning | Colab A100, free tier | 20 minutes | $0 |
| Debugging, failed runs, storage | various | n/a | $19 |
| Total | | | ~$60 |

Roughly a third of the budget went to mistakes. That number is included on purpose,
because it is a real part of what this kind of project costs.

## What went wrong

A gitignore rule silently excluded source code. The pattern data/ matched both the
intended output directory and src/pebblegpt/data/, so five pipeline files were never
committed. Anchoring the rule as /data/ fixes it.

A leftover restriction capped one data source at 23 percent of target. A data_files
argument added during small-scale testing survived into the full run, producing 986M
tokens against a 4.25B target with no error raised. The fix was a warning whenever
any source produces under 95 percent of its target.

Batched tokenization overshot targets by up to 2,400 percent. Switching from
per-document to batched tokenizer calls roughly doubled throughput but introduced a
bug where the token budget check only ran after a full batch completed. Caught on a
one million token local test rather than after hours of preparation.

RoPE tables were silently zeroed on model load. Registering the cosine and sine
tables as non-persistent buffers meant they were absent from the state dictionary,
and the fast initialization path in from_pretrained zero-filled them, turning rotary
position embedding into the identity function. The model still generated fluent
English with no positional information whatsoever. Caught only by comparing logits
against a directly constructed model. The fix is to compute RoPE lazily from
configuration values and never store it as a buffer.

A MIG GPU slice ran out of memory at batch size 16. A listing advertising 24GB VRAM
turned out to be a one seventh compute and memory partition of a larger card.

Data preparation on an H100 wasted roughly $22. Tokenization is CPU bound and single
threaded by default. Moving to a 32 vCPU instance cut the cost to $6.

The pattern across all of these: validate at a scale where mistakes are free, then
scale up.

## Acknowledgements

Architecture and training decisions follow Hugging Face's Smol Training Playbook and
the SmolLM2 paper. Training data from FineWeb-Edu, DCLM-Edu, Python-Edu, FineMath,
Cosmopedia, and SmolTalk. Evaluation with EleutherAI's lm-evaluation-harness.

## License

Apache 2.0