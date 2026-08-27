# PebbleGPT

An exploratory study in training a language model from scratch on a small budget.

The question: how much of a real language model pipeline can one person build and
run alone, following the published methodology of a real lab, on a small budget?

This repository contains the full pipeline: architecture, data curation,
pretraining, mid-training, supervised fine-tuning, and evaluation. Every component
was written from scratch rather than adapted from an existing training framework.

Models:
[PebbleGPT-320M](https://huggingface.co/strectelite/PebbleGPT-320M) (base),
[PebbleGPT-320M-Instruct](https://huggingface.co/strectelite/PebbleGPT-320M-Instruct)
(chat)

## Contents

1. The Story
2. Quick start
3. Results
4. Architecture
5. Repository structure
6. Reproducing
7. Compute cost
8. What went wrong
9. Acknowledgements
10. License

## The Story

This model started as a straightforward question: how much of a real language model
pipeline can one person build and run alone, following the published methodology of
a real lab, on a small budget?

The starting point was Hugging Face's Smol Training Playbook, the account of how
SmolLM3 was built. Rather than improvise architecture and training decisions, this
project borrowed them directly wherever the Playbook had already run the ablation.
Grouped query attention at ratio 4, tied embeddings, SwiGLU, no QK-norm, no Z-loss,
embeddings excluded from weight decay, the AdamW hyperparameter triplet that has gone
unchanged from Llama 1 through DeepSeek V3, and a Warmup-Stable-Decay learning rate
schedule. The data mixture ratios (FineWeb-Edu and DCLM-Edu split 50/50, code capped
at 10 percent, math at 5 percent) and the evaluation suite (HellaSwag, PIQA, ARC-easy,
with the noisier benchmarks the Playbook itself flagged as unreliable dropped) came
from the same source. The one deliberate departure was the tokenizer: SmolLM3 uses a
128k vocabulary that makes sense at 3B parameters, but at 320M it would consume 37
percent of the parameter budget as a lookup table, so the smaller 49k SmolLM2
tokenizer was used instead.

Pretraining ran for 10B tokens on a single rented H100. The result was a functioning
base model with loss 2.507 and benchmark scores in the expected range for a model this
small and this undertrained relative to modern practice.

At that point the model was evaluated against a set of comparable public models,
matched as closely as possible on parameter count and training scale. Two benchmarks
stood out as places where the model was underperforming what the token budget should
have allowed: PIQA, which measures physical commonsense reasoning, and ARC-easy, which
measures grade school science reasoning. Rather than accept that gap or retrain from
scratch, the model went through a short mid-training pass (an annealing phase in the
Playbook's terminology) aimed specifically at those two benchmarks.

The mixture for that phase was 40 percent replay of the original pretraining data,
mixed with Cosmopedia's `wikihow` and `stories` splits for PIQA and its `openstax` and
`khanacademy` splits for ARC-easy. The run was a warm restart from the fully decayed
pretraining checkpoint, 300M tokens, 30 minutes, on a rented RTX PRO 6000. It was the
single highest return step in the entire project: HellaSwag rose from 30.5 to 40.4,
PIQA from 55.5 to 67.7, and ARC-easy from 37.7 to 56.4.

Supervised fine-tuning followed, using SmolTalk with ChatML formatting and loss
masked to assistant response tokens only. This phase was free. As a student with
Colab Pro access, the fine-tuning run used a Colab-provisioned A100 rather than a
rented one, and took about 20 minutes for 50,000 conversations across 2 epochs.

## Quick start

```python
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
```

`trust_remote_code=True` is required. There is currently no GGUF conversion, so this
model does not run in llama.cpp or Ollama.

## Results

All scores are zero-shot, full test sets, measured with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The
comparison models were evaluated with the identical harness, task versions, and
batch size, rather than taken from other papers, so the numbers are directly
comparable.

### Across this model's own training phases

| Benchmark | After pretraining | After mid-training | After SFT |
|---|---|---|---|
| HellaSwag (acc_norm) | 30.5 | 40.4 | 40.60 |
| PIQA (acc_norm) | 55.5 | 67.7 | 66.76 |
| ARC-easy (acc_norm) | 37.7 | 56.4 | 53.79 |
| ARC-challenge (acc_norm) | n/a | 31.1 | 29.95 |
| Training loss | 2.507 | 2.067 | n/a |

Mid-training produced most of the movement. Supervised fine-tuning held those gains
within one to three points, the ordinary cost of instruction tuning rather than a
sign that anything went wrong.

### Against comparable public models

| Model | Params | Training tokens | HellaSwag | PIQA | ARC-easy | ARC-challenge |
|---|---|---|---|---|---|---|
| GPT-2 | 124M | ~8-10B | 31.14 | 62.51 | 39.48 | 22.70 |
| Pythia-160M | 160M | 300B | 30.15 | 61.70 | 39.56 | 23.81 |
| Pythia-410M | 410M | 300B | 40.61 | 66.97 | 45.92 | 24.40 |
| SmolLM2-135M | 135M | 2T | 43.10 | 68.34 | 58.54 | 29.86 |
| **PebbleGPT-320M-Instruct** | **320M** | **10.4B** | **40.60** | **66.76** | **53.79** | **29.95** |
| SmolLM2-360M | 360M | 4T | 56.36 | 72.03 | 68.01 | 38.05 |
| Qwen3-0.6B-Base | 600M | 36T | 53.83 | 70.02 | 57.91 | 38.40 |

The closest comparison on training budget is GPT-2, which used a similar token count
to this model. At nearly identical data scale, this model is ahead on both ARC
benchmarks and roughly level on HellaSwag and PIQA. Against models trained on 30 to
3,500 times more data, the gap is real and is the honest limit of a 10.4B token
budget, not something the mid-training pass could fully close.

### Instruction following

| Metric | Score |
|---|---|
| IFEval inst-level loose | 27.8 |
| IFEval inst-level strict | 27.2 |
| IFEval prompt-level loose | 14.4 |
| IFEval prompt-level strict | 13.7 |

## Architecture

Dense decoder-only transformer.

| Parameter | Value |
|---|---|
| Total parameters | 320.9M |
| Non-embedding parameters | 270.6M |
| Layers | 24 |
| Hidden size | 1024 |
| Intermediate size | 2816 |
| Attention | Grouped query attention, 16 query heads, 4 KV heads |
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
the parameter count comparable, and matches SmolLM2-135M and SmolLM2-360M at this
scale.

Vocabulary of 49,152, not 128k. At 320M parameters a 128k vocabulary would put 37
percent of the budget into a lookup table. The smaller vocabulary reduces that to 14
percent, freeing roughly 80M parameters for transformer layers.

Grouped query attention at ratio 4. Sixteen query heads share four KV heads, cutting
the KV cache to a quarter of full multi-head attention with no measured quality loss
at this scale.

Warmup-Stable-Decay learning rate schedule. Unlike cosine decay, WSD does not require
committing to a total token count before training starts. The learning rate holds at
peak for most of the run and decays only in the final stretch.

### Pretraining

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

### Mid-training

300M tokens, warm restart from the fully decayed pretraining checkpoint with a fresh
optimizer state. Peak learning rate 1.2e-4, roughly 25 percent of the original peak,
30 warmup steps, 85 percent decay window, 572 steps.

| Source | Share | Purpose |
|---|---|---|
| Baseline replay from pretraining mixture | 40% | Prevent narrowing |
| Cosmopedia wikihow | 20% | Procedural and physical reasoning |
| Cosmopedia stories | 10% | Everyday world knowledge |
| Cosmopedia openstax | 15% | Science curriculum |
| Cosmopedia khanacademy | 15% | Science curriculum |

Cosmopedia is synthetic data generated by Mixtral-8x7B-Instruct-v0.1, decontaminated
by its authors against ARC, PIQA, HellaSwag, OpenBookQA, WinoGrande, MMLU, and BoolQ
using 10-gram overlap detection.

One RTX PRO 6000, 30 minutes. Final loss 2.067.

### Supervised fine-tuning

50,000 conversations from SmolTalk, 2 epochs, ChatML format. Loss computed only on
assistant response tokens, with user turns, template tokens, and padding masked with
-100. Getting this masking exactly right matters: a subtle bug here produces a
plausible looking but useless model, so it has dedicated unit tests.

AdamW at 2e-5, weight decay 0.01, OneCycleLR cosine schedule, 3 percent warmup,
batch size 32, sequence length 1,024, 3,126 steps.

One A100 80GB, Colab Pro, 20 minutes.

## Repository structure

**`src/pebblegpt/model/`**

| File | Contents |
|---|---|
| `attention.py` | Grouped query attention with RoPE and KV cache |
| `SwiGLU.py` | Gated feedforward layer |
| `block.py` | Transformer block, pre-norm |
| `model.py` | PebbleGPT, the training-time model |
| `modeling.py` | HuggingFace-compatible wrapper with generation |
| `configuration.py` | HuggingFace config class |

**`src/pebblegpt/data/`**

| File | Contents |
|---|---|
| `download.py` | Source registry and mixture ratios |
| `filter.py` | N-gram repetition filter |
| `tokenize.py` | Batched tokenization, packing, sharding |
| `shuffle.py` | Sequence-level shuffle across sources |
| `loader.py` | Memory-mapped dataset |
| `sft_dataset.py` | ChatML formatting and loss masking |
| `anneal_sources.py` | Mid-training data sources |

**`src/pebblegpt/train/`**

| File | Contents |
|---|---|
| `scheduler.py` | Warmup-Stable-Decay |
| `optimizer.py` | AdamW with weight decay exclusions |
| `loop.py` | Training loop, gradient accumulation, MFU tracking |

**`src/pebblegpt/eval/`**

| File | Contents |
|---|---|
| `harness.py` | Export to HF format, run lm-evaluation-harness |

**`src/pebblegpt/utils/`**

| File | Contents |
|---|---|
| `checkpoint.py` | Atomic saves with full resume state |
| `logging.py` | Console, JSONL, and Weights and Biases |

**`src/scripts/`**

| File | Contents |
|---|---|
| `prepare_data.py` | Pretraining data pipeline |
| `prepare_anneal_data.py` | Mid-training data pipeline |
| `train.py` | Pretraining and mid-training |
| `train_sft.py` | Supervised fine-tuning |
| `benchmark_sft.py` | Batch size and sequence length sweep |
| `evaluate.py` | Export and evaluate a checkpoint |
| `inspect_data.py` | Decode packed records for review |
| `diagnose_filter.py` | Break down filter rejection reasons |

## Reproducing

```bash
git clone https://github.com/sanjayriram44/PebbleGPT.git
cd PebbleGPT
uv sync

cat > .env <<'EOF'
HF_TOKEN=hf_...
WANDB_API_KEY=...
WANDB_ENTITY=your_entity
EOF
```

Data preparation. Run this on a CPU instance, not a GPU.

```bash
uv run python src/scripts/prepare_data.py --total-tokens 10_000_000_000
```

Pretraining.

```bash
uv run python src/scripts/train.py \
    --micro-batch-size 16 --grad-accum-steps 16 \
    --total-tokens 10_000_000_000 --eval-every 1000 --wandb
```

Mid-training.

```bash
uv run python src/scripts/prepare_anneal_data.py --total-tokens 300_000_000

uv run python src/scripts/train.py \
    --data-dir data/anneal_train --ckpt-dir checkpoints/anneal \
    --anneal-from checkpoints/ckpt_final.pt --anneal-peak-lr 1.2e-4 \
    --total-tokens 300_000_000 --warmup-steps 30 --decay-fraction 0.85 \
    --micro-batch-size 16 --grad-accum-steps 16 --no-resume --wandb
```

Supervised fine-tuning. Fits comfortably in a free Colab session.

```bash
uv run python src/scripts/benchmark_sft.py

uv run python src/scripts/train_sft.py \
    --batch-size 32 --seq-len 1024 --n-convos 50000 --epochs 2 \
    --push-to-hub your-username/PebbleGPT-320M-Instruct
```

Evaluation.

```bash
uv run python src/scripts/evaluate.py \
    --ckpt checkpoints/anneal/ckpt_final.pt \
    --tasks hellaswag piqa arc_easy arc_challenge --device cuda
```

## Compute cost

| Phase | Hardware | Duration | Cost |
|---|---|---|---|
| Pretraining, including data preparation and debugging | H100 SXM, plus a 32 vCPU instance for data prep | ~24 hours combined | $50 |
| Mid-training, including the data pipeline for it | RTX PRO 6000 | ~30 minutes | $1 |
| Supervised fine-tuning | A100, Colab Pro | ~20 minutes | $0 |
| **Total** | | | **~$60** |

Roughly a third of the pretraining figure was spent on debugging rather than clean
training time. That cost is included rather than hidden, because it is a real part of
what a first solo attempt at this actually takes.

## What went wrong

The failures below are not covered in any published training guide. They are the
specific things that break when one person does this alone for the first time.

A gitignore rule silently excluded source code. The pattern `data/` matched both the
intended output directory and `src/pebblegpt/data/`, so five pipeline files were never
committed. This surfaced only when a fresh clone on a rented GPU could not import
them. Anchoring the rule as `/data/` fixes it.

A leftover restriction capped one data source at 23 percent of target. A `data_files`
argument added during small-scale testing survived into the full run, producing 986M
tokens against a 4.25B target with no error raised. The fix was a warning whenever any
source produces under 95 percent of its target.

Batched tokenization overshot targets by up to 2,400 percent. Switching from
per-document to batched tokenizer calls roughly doubled throughput but introduced a
bug where the token budget check only ran after a full batch completed. Caught on a
one million token local test rather than after hours of preparation.

RoPE tables were silently zeroed on model load. Registering the cosine and sine
tables as non-persistent buffers meant they were absent from the state dictionary,
and the fast initialization path in `from_pretrained` zero-filled them, turning
rotary position embedding into the identity function. The model still generated
fluent English with no positional information whatsoever. Caught only by comparing
logits against a directly constructed model. The fix is to compute RoPE lazily from
configuration values and never store it as a buffer.

A MIG GPU slice ran out of memory at batch size 16. A listing advertising 24GB VRAM
turned out to be a one seventh compute and memory partition of a larger card.

Data preparation on an H100 wasted roughly $22 before being moved to a 32 vCPU
instance. Tokenization is CPU bound and single threaded by default, and the GPU sat
at 1 percent utilization for hours before the fix.

The pattern across all of these: validate at a scale where mistakes are free, then
scale up.

## Acknowledgements

Architecture and training decisions follow Hugging Face's
[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
and the [SmolLM2 paper](https://arxiv.org/abs/2502.02737).

Training data from FineWeb-Edu, DCLM-Edu, Python-Edu, FineMath, Cosmopedia, and
SmolTalk. Evaluation with EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## License

Apache 2.0
