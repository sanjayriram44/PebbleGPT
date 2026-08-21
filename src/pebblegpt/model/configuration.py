"""HF-compatible config for PebbleGPT.

Defaults mirror PebbleGPT's so a training checkpoint loads without remapping.
"""

from transformers import PretrainedConfig


class PebbleGPTConfig(PretrainedConfig):
    model_type = "pebblegpt"

    def __init__(self,
                 vocab_size: int = 49152,
                 hidden_size: int = 1024,
                 num_hidden_layers: int = 24,
                 num_heads: int = 16,
                 num_kv_heads: int = 4,
                 intermediate_size: int = 2816,
                 max_seq_len: int = 2048,
                 rope_base: float = 10000.0,
                 norm_eps: float = 1e-6,
                 tie_word_embeddings: bool = True,
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.norm_eps = norm_eps

        # Aliases some HF tooling looks for.
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.max_position_embeddings = max_seq_len
        self.rope_theta = rope_base
        self.rms_norm_eps = norm_eps

        # `pebblegpt` isn't a registered transformers architecture, so exports
        # carry their own source and load via trust_remote_code.
        self.auto_map = {
            "AutoConfig": "configuration.PebbleGPTConfig",
            "AutoModelForCausalLM": "modeling.PebbleGPTForCausalLM",
        }

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)