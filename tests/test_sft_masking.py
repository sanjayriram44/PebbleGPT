"""Verify loss masking hits exactly the assistant response tokens."""

from transformers import AutoTokenizer

from pebblegpt.data.sft_dataset import build_masked_example, IGNORE_INDEX

TOKENIZER = "HuggingFaceTB/SmolLM2-360M"


def test_masks_user_turn_keeps_assistant_turn():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]
    ex = build_masked_example(messages, tok, max_len=64)

    # decode only the unmasked (label != -100) positions
    kept_ids = [tid.item() for tid, lab in zip(ex.input_ids, ex.labels)
                if lab.item() != IGNORE_INDEX]
    kept_text = tok.decode(kept_ids)

    print("KEPT TEXT:", repr(kept_text))

    assert "2+2 equals 4" in kept_text, "assistant response should be unmasked"
    assert "What is 2+2?" not in kept_text, "user turn should be masked out"


def test_padding_is_masked():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    ex = build_masked_example(messages, tok, max_len=256)  # forces real padding

    # attention_mask==0 positions must have labels==IGNORE_INDEX
    for am, lab in zip(ex.attention_mask, ex.labels):
        if am.item() == 0:
            assert lab.item() == IGNORE_INDEX


def test_label_token_count_matches_assistant_only():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    messages = [
        {"role": "user", "content": "Tell me a fact."},
        {"role": "assistant", "content": "Water boils at 100 degrees Celsius at sea level."},
    ]
    ex = build_masked_example(messages, tok, max_len=128)

    n_kept = (ex.labels != IGNORE_INDEX).sum().item()
    assistant_only_ids = tok.encode(
        f"<|im_start|>assistant\nWater boils at 100 degrees Celsius at sea level.<|im_end|>\n",
        add_special_tokens=False,
    )
    assert n_kept == len(assistant_only_ids), \
        f"expected {len(assistant_only_ids)} unmasked tokens, got {n_kept}"


if __name__ == "__main__":
    test_masks_user_turn_keeps_assistant_turn()
    test_padding_is_masked()
    test_label_token_count_matches_assistant_only()
    print("all masking tests passed")