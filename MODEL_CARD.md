---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- pebblegpt
- small-language-model
base_model: strectelite/PebbleGPT-320M
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/dclm-edu
- HuggingFaceTB/finemath
- HuggingFaceTB/cosmopedia
- HuggingFaceTB/smoltalk
---

# PebbleGPT-320M-Instruct

## Table of Contents

1. Model Summary
2. Usage
3. Evaluation
4. Training
5. Limitations
6. License
7. Citation

## Model Summary

PebbleGPT-320M-Instruct is a 320M parameter language model trained from scratch as an
exploratory study. The question behind it: how far can one person get on roughly $60
of rented compute, writing the architecture by hand and curating the training data
themselves?

The result is a model that holds a coherent conversation, writes working Python, and
scores well above chance on standard commonsense and science benchmarks, while
remaining unreliable on facts.

Training ran in three phases: pretraining on 10B tokens of curated web, code, and math
data, a 300M token annealing phase targeted at physical and scientific reasoning, and
supervised fine-tuning on 50,000 conversations.

- Architecture: dense decoder-only transformer, 24 layers, hidden size 1024
- Attention: grouped query attention, 16 query heads, 4 KV heads
- Context: 2048 tokens
- Tokenizer: SmolLM2, 49,152 vocabulary
- Training tokens: 10.4B across three phases
- Compute cost: approximately $60

## Usage

pip install transformers torch

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = "strectelite/PebbleGPT-320M-Instruct"
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True
    ).to(device)

    prompt = "Explain what recursion is in simple terms."
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_k=40,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

trust_remote_code=True is required. The architecture is not registered in
transformers, so the model source ships alongside the weights.

### Chat format

The model uses ChatML.

    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>

### Sample outputs

Write a Python function that reverses a string.

    def reverse_string(s):
        return s[::-1]

Explain what recursion is in simple terms.

Recursion is a technique in programming to solve a problem by breaking it down into
smaller, simpler parts, each of which can be solved independently.

Summarize why the sky is blue in one sentence.

The sky is blue because it has a rich blue color.

The third example shows a common failure mode: fluent, correctly formatted, and
substantively empty.

## Evaluation

Zero-shot, full test sets, evaluated with lm-evaluation-harness.

| Benchmark | Metric | Score |
|---|---|---|
| HellaSwag | acc_norm | 40.6 |
| PIQA | acc_norm | 66.8 |
| ARC-easy | acc_norm | 53.8 |
| ARC-challenge | acc_norm | 30.0 |
| IFEval | inst-level loose | 27.8 |
| IFEval | inst-level strict | 27.2 |
| IFEval | prompt-level loose | 14.4 |
| IFEval | prompt-level strict | 13.7 |

### Scores across training phases

| Benchmark | After pretraining | After annealing | After SFT |
|---|---|---|---|
| HellaSwag | 30.5 | 40.4 | 40.6 |
| PIQA | 55.5 | 67.7 | 66.8 |
| ARC-easy | 37.7 | 56.4 | 53.8 |
| ARC-challenge | n/a | 31.1 | 30.0 |
| Training loss | 2.507 | 2.067 | n/a |

MMLU, GSM8K, and HumanEval are not reported. At this parameter count and token budget
they sit at or near chance.

## Training

### Architecture

| Parameter | Value |
|---|---|
| Total parameters | 320.9M |
| Non-embedding parameters | 270.6M |
| Layers | 24 |
| Hidden size | 1024 |
| Intermediate size | 2816 |
| Attention heads | 16 |
| KV heads | 4 |
| Head dimension | 64 |
| Activation | SwiGLU |
| Normalization | RMSNorm, pre-norm |
| Positional encoding | RoPE, theta 10000 |
| Embeddings | Tied input and output |
| Context length | 2048 |
| Vocabulary size | 49,152 |
| Precision | bfloat16 mixed precision |

The intermediate size of 2816 is roughly 8/3 of the hidden size rather than the more
common 4x. SwiGLU uses three weight matrices where a standard feedforward layer uses
two, so the 8/3 ratio keeps the parameter count comparable to a non-gated layer at 4x
width.

The 49,152 token vocabulary was chosen over larger alternatives because embedding
tables do not shrink with model size. At 320M parameters a 128k vocabulary would
consume 37 percent of the parameter budget as a lookup table. The smaller vocabulary
reduces that to 14 percent.

### Phase 1: Pretraining

10B tokens, 19,073 steps at 524,288 tokens per step.

| Source | Share | Tokens |
|---|---|---|
| FineWeb-Edu | 42.5% | 4.25B |
| DCLM-Edu, filtered to edu_int_score >= 3 | 42.5% | 4.25B |
| Python-Edu | 10% | 1.0B |
| FineMath-4+ | 5% | 0.5B |

AdamW, betas 0.9 and 0.95, eps 1e-8. Weight decay 0.1 excluding embeddings and norms.
Gradient clipping 1.0. Warmup-Stable-Decay schedule, peak learning rate 5e-4, minimum
5e-5, 2,000 warmup steps, 10 percent decay window. Global batch 524,288 tokens.

Documents were filtered for n-gram repetition, tokenized, packed into fixed 2,049
token records with EOS separators, then shuffled at the sequence level.

Hardware: one H100 SXM, approximately 18 hours at 35 percent model FLOPs utilization.
Final training loss 2.507.

### Phase 2: Annealing

300M tokens of targeted continued pretraining, aimed at physical commonsense
reasoning and grade school science.

| Source | Share | Purpose |
|---|---|---|
| Baseline replay from phase 1 mixture | 40% | Prevent narrowing |
| Cosmopedia wikihow | 20% | Procedural and physical reasoning |
| Cosmopedia stories | 10% | Everyday world knowledge |
| Cosmopedia openstax | 15% | Science curriculum content |
| Cosmopedia khanacademy | 15% | Science curriculum content |

Warm restart from the fully decayed phase 1 checkpoint, fresh optimizer state. Peak
learning rate 1.2e-4, roughly 25 percent of the original peak. 30 warmup steps, 85
percent decay window, 572 steps total.

This is a warm restart from a fully decayed checkpoint rather than a conventional
annealing phase applied to a pre-decay checkpoint.

Cosmopedia is synthetic data generated by Mixtral-8x7B-Instruct-v0.1. Its authors
decontaminated it against ARC, PIQA, HellaSwag, OpenBookQA, WinoGrande, MMLU, and
BoolQ using 10-gram overlap detection.

Hardware: one RTX PRO 6000, approximately 30 minutes. Final training loss 2.067.

### Phase 3: Supervised fine-tuning

50,000 conversations from SmolTalk, 2 epochs.

Format: ChatML. Loss masking: assistant response tokens only, with -100 on user
turns, template tokens, and padding. Optimizer: AdamW, weight decay 0.01, learning
rate 2e-5, OneCycleLR cosine schedule, 3 percent warmup. Batch size 32, sequence
length 1,024, 3,126 steps.

Hardware: one A100 80GB, approximately 20 minutes.

### Compute

| Phase | Hardware | Duration | Cost |
|---|---|---|---|
| Data preparation | 32 vCPU instance | 5 hours | $6 |
| Pretraining | H100 SXM | 18 hours | $34 |
| Annealing | RTX PRO 6000 | 30 minutes | $1 |
| Supervised fine-tuning | A100 80GB | 20 minutes | $0 |
| Debugging and failed runs | various | n/a | $19 |
| Total | | | ~$60 |

### Software

PyTorch, transformers, datasets, and lm-evaluation-harness. The model, training loop,
data pipeline, learning rate schedule, and checkpointing were written from scratch.

## Limitations

Factual reliability is poor. Asked for the capital of France, the model correctly
answers Paris and then places it in a charming town called Marseille. A 10B token
budget does not encode reliable world knowledge at this parameter count.

Instruction following is partial. An IFEval prompt-level strict score of 13.7 means
the model satisfies every constraint in a prompt roughly one time in seven.

Benchmark gains are concentrated. PIQA and ARC-easy improved most because the
annealing data was selected to target them. This is a documented technique rather
than contamination, but it means the model is not uniformly stronger.

No safety alignment. Fine-tuned on SmolTalk only. No preference optimization, no red
teaming, no safety evaluation.

Short context. 2,048 tokens, with no long context extension.

Research and education only. Not suitable for production use, factual lookup, or any
application where being wrong has consequences.

## License

Apache 2.0

## Citation

    @software{pebblegpt2026,
      author = {Sriram, Sanjay},
      title = {PebbleGPT: an exploratory study in training a language model
               from scratch on a small budget},
      year = {2026},
      url = {https://huggingface.co/strectelite/PebbleGPT-320M-Instruct}
    }