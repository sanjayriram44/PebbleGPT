"""Run lm-evaluation-harness (EleutherAI) against a PebbleGPT checkpoint.

Training checkpoints are exported to a self-contained HF directory first —
weights, tokenizer, and model source — then evaluated. The harness is the de
facto standard; it's what populates the HF Open LLM Leaderboard.
"""

import subprocess
from pathlib import Path

from transformers import AutoTokenizer

from pebblegpt.data.tokenize import TOKENIZER
from pebblegpt.model.configuration import PebbleGPTConfig
from pebblegpt.model.modeling import from_training_checkpoint

# Core suite — the ones that actually move at 320M.
CORE_TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge"]

# Near-random at this scale; run occasionally, not every checkpoint.
EXTRA_TASKS = ["mmlu", "gsm8k"]

# Source files copied into the export so it loads standalone.
MODEL_SOURCES = ("configuration.py", "modeling.py", "block.py",
                 "attention.py", "SwiGLU.py")


def export_hf_model(ckpt_path: Path,
                    out_dir: Path,
                    config: PebbleGPTConfig | None = None) -> Path:
    """Convert a training checkpoint into a self-contained HF directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = from_training_checkpoint(ckpt_path, config=config)
    model.save_pretrained(out_dir)

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.save_pretrained(out_dir)

    # Copy model source, rewriting package imports to relative ones — HF loads
    # remote code as a package, and the files sit flat in the export dir.
    model_pkg = Path(__file__).parent.parent / "model"
    for name in MODEL_SOURCES:
        src = (model_pkg / name).read_text()
        src = src.replace("from pebblegpt.model.", "from .")
        (out_dir / name).write_text(src)

    print(f"exported -> {out_dir}")
    return out_dir


def run_eval(model_dir: Path,
             tasks: list[str] | None = None,
             output_dir: Path = Path("eval_results"),
             device: str = "cuda",
             batch_size: str = "auto",
             num_fewshot: int | None = None,
             limit: int | None = None) -> None:
    """Invoke lm_eval as a subprocess."""
    tasks = tasks or CORE_TASKS

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True",
        "--tasks", ",".join(tasks),
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", str(output_dir),
        "--trust_remote_code",
    ]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        # subsample for speed; the Playbook used 1,000 questions per benchmark
        cmd += ["--limit", str(limit)]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)