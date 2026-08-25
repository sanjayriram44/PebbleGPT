"""Additional sources for the PIQA/ARC-easy annealing experiment.

Cosmopedia v2 splits, chosen for topical proximity to the target evals:
- wikihow/stories  -> PIQA  (step-by-step procedural, everyday-world knowledge)
- openstax/khanacademy -> ARC-easy (grade-school/course-outline science)

This is genuinely uncertain: Cosmopedia is synthetic (Mixtral-generated)
textbook-style text, not naturalistic PIQA/ARC-like text, and we're annealing
from a FULLY DECAYED checkpoint rather than the pre-decay checkpoint every
documented annealing example uses. Treat the outcome as an experiment result,
not a guaranteed win.
"""

from dataclasses import dataclass
from datasets import load_dataset

ANNEAL_MIXTURE = {
    "baseline_replay": 0.40,   # existing fineweb-edu/dclm-edu/code/finemath shards
    "piqa_wikihow":     0.20,
    "piqa_stories":      0.10,
    "arc_openstax":      0.15,
    "arc_khanacademy":   0.15,
}


@dataclass
class AnnealSourceSpec:
    repo: str
    config: str
    text_col: str = "text"


ANNEAL_SOURCES: dict[str, AnnealSourceSpec] = {
    "piqa_wikihow": AnnealSourceSpec(
        repo="HuggingFaceTB/cosmopedia",   # was cosmopedia-v2
        config="wikihow",
    ),
    "piqa_stories": AnnealSourceSpec(
        repo="HuggingFaceTB/cosmopedia",
        config="stories",
    ),
    "arc_openstax": AnnealSourceSpec(
        repo="HuggingFaceTB/cosmopedia",
        config="openstax",
    ),
    "arc_khanacademy": AnnealSourceSpec(
        repo="HuggingFaceTB/cosmopedia",
        config="khanacademy",
    ),
}


def anneal_token_budget(total_tokens: int) -> dict[str, int]:
    """Target tokens per anneal-only source (excludes baseline_replay,
    which is handled separately by sampling existing shards)."""
    return {
        name: int(total_tokens * ANNEAL_MIXTURE[name])
        for name in ANNEAL_SOURCES
    }


def load_anneal_source(name: str, streaming: bool = True):
    spec = ANNEAL_SOURCES[name]
    ds = load_dataset(spec.repo, name=spec.config, split="train", streaming=streaming)
    return ds, spec.text_col