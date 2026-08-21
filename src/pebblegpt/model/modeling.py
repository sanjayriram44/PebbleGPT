"""HF PreTrainedModel wrapper around PebbleGPT's blocks.

Used for evaluation (LightEval) and Hub publishing. Training uses the plain
PebbleGPT class in model.py — attribute names match, so checkpoints load
into either without key remapping.

Generation is not supported yet: attention has no KV cache, so generate()
would be unusably slow. That's fine for the multiple-choice benchmarks
(HellaSwag, PIQA, ARC, MMLU), which only need forward passes.
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from pebblegpt.model.block import TransformerBlock
from pebblegpt.model.configuration import PebbleGPTConfig


class PebbleGPTForCausalLM(PreTrainedModel):
    config_class = PebbleGPTConfig
    base_model_prefix = "pebblegpt"
    supports_gradient_checkpointing = False
    _no_split_modules = ["TransformerBlock"]
    _tied_weights_keys = {"proj_head.weight": "token_embedding.weight"}

    def __init__(self, config: PebbleGPTConfig):
        super().__init__(config)

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                intermediate_size=config.intermediate_size,
                max_seq_len=config.max_seq_len,
                rope_base=config.rope_base,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.proj_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        if config.tie_word_embeddings:
            self.proj_head.weight = self.token_embedding.weight

    # --- HF plumbing -------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.proj_head

    def set_output_embeddings(self, value):
        self.proj_head = value
        
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.proj_head.weight = self.token_embedding.weight

    # --- forward -----------------------------------------------------

    def forward(self,
                input_ids: torch.LongTensor | None = None,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_values=None,
                inputs_embeds: torch.FloatTensor | None = None,
                labels: torch.LongTensor | None = None,
                use_cache: bool | None = None,
                output_attentions: bool | None = None,
                output_hidden_states: bool | None = None,
                return_dict: bool | None = None,
                **kwargs) -> CausalLMOutputWithPast:
        """Attention is always causal, so attention_mask / position_ids /
        past_key_values are accepted for interface compatibility and ignored."""
        return_dict = return_dict if return_dict is not None else True

        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = self.token_embedding(input_ids)
        else:
            raise ValueError("provide either input_ids or inputs_embeds")

        hidden_states = [] if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                hidden_states.append(x)
            x = block(x)

        x = self.final_norm(x)
        if output_hidden_states:
            hidden_states.append(x)

        logits = self.proj_head(x)

        loss = None
        if labels is not None:
            # HF convention: labels are unshifted, the model shifts internally
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            out = (logits,)
            if output_hidden_states:
                out += (tuple(hidden_states),)
            return ((loss,) + out) if loss is not None else out

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=None,
        )


def from_training_checkpoint(ckpt_path,
                             config: PebbleGPTConfig | None = None,
                             device: str = "cpu") -> PebbleGPTForCausalLM:
    """Load a PebbleGPT training checkpoint into the HF wrapper."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PebbleGPTForCausalLM(config or PebbleGPTConfig())
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing:
        print(f"missing keys: {missing}")
    if unexpected:
        print(f"unexpected keys: {unexpected}")
    return model.to(device).eval()