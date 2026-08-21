"""Run lm-evaluation-harness (EleutherAI) against a PebbleGPT checkpoint.

Training checkpoints are exported to a self-contained HF directory first —
weights, tokenizer, and model source — then evaluated. The harness is the de
facto standard; it's what populates the HF Open LLM Leaderboard.
"""

import json
import subprocess
import tempfile
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

# lm-eval returns keys like "acc,none" and "acc_stderr,none"; drop the
# suffix and skip error bars and aliases.
_SKIP_METRICS = {"alias", "sample_len"}


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


def _build_cmd(model_dir: Path,
               tasks: list[str],
               device: str,
               batch_size: str,
               output_path: str,
               num_fewshot: int | None,
               limit: int | None) -> list[str]:
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},trust_remote_code=True",
        "--tasks", ",".join(tasks),
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", output_path,
        "--trust_remote_code",
    ]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        # subsample for speed; the Playbook used 1,000 questions per benchmark
        cmd += ["--limit", str(limit)]
    return cmd


def _parse_results(results_dir: str) -> dict[str, float]:
    """Flatten lm-eval's JSON output into {task_metric: value}."""
    out: dict[str, float] = {}

    for path in Path(results_dir).rglob("results_*.json"):
        data = json.loads(path.read_text())
        for task, metrics in data.get("results", {}).items():
            for metric, value in metrics.items():
                base = metric.split(",")[0]        # "acc,none" -> "acc"
                if base in _SKIP_METRICS or base.endswith("_stderr"):
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                out[f"{task}_{base}"] = float(value)

    return out


def run_eval_inline(model_dir: Path,
                    tasks: list[str] | None = None,
                    device: str = "cuda",
                    batch_size: str = "auto",
                    limit: int | None = 1000) -> dict[str, float]:
    """Run lm_eval and return {task_metric: value}.

    Runs as a subprocess so a harness crash can't take down training. Failures
    raise with the tail of stderr attached — a bare exit code tells you nothing
    useful hours into a run.
    """
    tasks = tasks or CORE_TASKS

    with tempfile.TemporaryDirectory() as tmp:
        cmd = _build_cmd(model_dir, tasks, device, batch_size, tmp, None, limit)
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-20:]
            raise RuntimeError("lm_eval failed:\n" + "\n".join(tail))

        return _parse_results(tmp)


def run_eval(model_dir: Path,
             tasks: list[str] | None = None,
             output_dir: Path = Path("eval_results"),
             device: str = "cuda",
             batch_size: str = "auto",
             num_fewshot: int | None = None,
             limit: int | None = None) -> None:
    """Standalone eval — streams output to the terminal."""
    tasks = tasks or CORE_TASKS
    cmd = _build_cmd(model_dir, tasks, device, batch_size,
                     str(output_dir), num_fewshot, limit)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)