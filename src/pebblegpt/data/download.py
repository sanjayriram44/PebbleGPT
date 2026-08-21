"""Dataset loading for PebbleGPT pretraining.

Mixture follows the SmolLM3 Playbook's validated stage-1 split, English-only:
    ~85% web  (FineWeb-Edu + DCLM-Edu, 50/50)
    ~10% code (Python-Edu)
    ~5%  math (FineMath-4+)
"""

from dataclasses import dataclass
from datasets import load_dataset

# Chinchilla-optimal for a 320M model (~20 tokens/param).
CHINCHILLA_TOKENS = 6_400_000_000

MIXTURE = {
    "fineweb-edu": 0.425,
    "dclm-edu":    0.425,
    "code":        0.10,
    "finemath":    0.05,
}


@dataclass
class SourceSpec:
    repo: str
    config: str | None
    text_col: str
    min_edu_score: int | None = None
    data_files: str | list[str] | None = None
    data_dir: str | None = None


SOURCES: dict[str, SourceSpec] = {
    "fineweb-edu": SourceSpec(
        repo="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        text_col="text",
    ),
    "dclm-edu": SourceSpec(
        repo="HuggingFaceTB/dclm-edu",
        config=None,
        text_col="text",
        min_edu_score=3,
    ),
    "code": SourceSpec(
        # Python-Edu with S3 contents already hydrated into a `text` column.
        # The official HuggingFaceTB/stack-edu and smollm-corpus/python-edu
        # ship only blob_id references and require AWS credentials plus one
        # S3 request per file (~6 hours on a 16-core AWS instance).
        repo="Avelina/python-edu-cleaned",
        config=None,
        text_col="text",
    ),
    "finemath": SourceSpec(
        repo="HuggingFaceTB/finemath",
        config="finemath-4plus",
        text_col="text",
    ),
}


def token_budget(total_tokens: int = CHINCHILLA_TOKENS) -> dict[str, int]:
    """Target token count per source for a given total budget."""
    return {name: int(total_tokens * frac) for name, frac in MIXTURE.items()}


def load_source(name: str, streaming: bool = True, limit: int | None = None):
    """Load one source. Returns (dataset, text_column).

    Streaming is the default: combined with early-stopping in tokenize.py,
    it means you never consume more than the token budget requires. For large
    budgets or flaky connections, streaming=False uses the resumable download
    path instead, which is generally better for the real run.
    """
    spec = SOURCES[name]

    kwargs = {}
    if spec.data_files is not None:
        kwargs["data_files"] = spec.data_files
    if spec.data_dir is not None:
        kwargs["data_dir"] = spec.data_dir

    ds = load_dataset(
        spec.repo,
        name=spec.config,
        split="train",
        streaming=streaming,
        **kwargs,
    )

    if spec.min_edu_score is not None:
        ds = ds.filter(lambda x: x.get("edu_int_score", 0) >= spec.min_edu_score)

    if limit is not None:
        ds = ds.take(limit) if streaming else ds.select(range(limit))

    return ds, spec.text_col