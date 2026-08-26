"""SFT data: ChatML formatting + loss masking.

Only assistant response tokens contribute to loss. Everything else — the
ChatML template tokens, the user turn, the role labels — is masked with -100,
which torch.nn.functional.cross_entropy's default ignore_index skips.

Unlike pretraining's PackedDataset, SFT does NOT pack multiple conversations
into one sequence — mixing conversations would let the model attend across
unrelated exchanges, and masking would need document boundaries within a
single training example, which pretraining's packing wasn't designed for.
Each conversation is padded or truncated to a fixed length instead.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


@dataclass
class ChatMLExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


def format_chatml(messages: list[dict], tokenizer: PreTrainedTokenizerBase) -> str:
    """messages: [{"role": "user", "content": "..."}, {"role": "assistant", ...}]"""
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    return "".join(parts)


def build_masked_example(messages: list[dict],
                         tokenizer: PreTrainedTokenizerBase,
                         max_len: int = 1024) -> ChatMLExample:
    """Tokenize a full ChatML conversation and mask everything except
    assistant response spans.

    Masking works by tokenizing incrementally, turn by turn: after each
    non-assistant turn we know how many tokens to mask; after each assistant
    turn we know how many tokens to keep. This avoids needing to search for
    substring boundaries in token-id space, which is fragile.
    """
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert im_start is not None and im_start != tokenizer.unk_token_id, \
        "<|im_start|> not in tokenizer vocab"
    assert im_end is not None and im_end != tokenizer.unk_token_id, \
        "<|im_end|> not in tokenizer vocab"

    all_ids: list[int] = []
    all_labels: list[int] = []

    for m in messages:
        turn_text = f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)

        all_ids.extend(turn_ids)

        if m["role"] == "assistant":
            all_labels.extend(turn_ids)          # keep — this is what SFT learns
        else:
            all_labels.extend([IGNORE_INDEX] * len(turn_ids))  # mask

    all_ids = all_ids[:max_len]
    all_labels = all_labels[:max_len]

    pad_len = max_len - len(all_ids)
    attention_mask = [1] * len(all_ids) + [0] * pad_len

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    all_ids = all_ids + [pad_id] * pad_len
    all_labels = all_labels + [IGNORE_INDEX] * pad_len   # never learn from padding

    return ChatMLExample(
        input_ids=torch.tensor(all_ids, dtype=torch.long),
        labels=torch.tensor(all_labels, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
    )


class SFTDataset(Dataset):
    def __init__(self, conversations: list[list[dict]],
                tokenizer: PreTrainedTokenizerBase, max_len: int = 1024):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict:
        ex = build_masked_example(self.conversations[idx], self.tokenizer, self.max_len)
        return {
            "input_ids": ex.input_ids,
            "labels": ex.labels,
            "attention_mask": ex.attention_mask,
        }


def load_smoltalk(n_examples: int = 20_000, seed: int = 1337) -> list[list[dict]]:
    """Load and normalize a slice of SmolTalk into [{"role":.., "content":..}] lists."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    out = []
    for row in ds:
        msgs = row.get("messages")
        if not msgs:
            continue
        # normalize to just role/content, dropping any extra fields
        out.append([{"role": m["role"], "content": m["content"]} for m in msgs])
        if len(out) >= n_examples:
            break
    return out