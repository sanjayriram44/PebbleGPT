"""Warmup-Stable-Decay learning rate schedule.

WSD holds the peak LR constant for most of training, then decays over the
final fraction. Unlike cosine, it doesn't require committing to a total step
count up front — you decide when to enter decay during the run.

The Playbook's ablation found WSD matches cosine's final performance: cosine
leads during WSD's stable phase, but WSD catches up during decay. Hägele et
al. (2024) recommend 10-20% of tokens for decay; SmolLM3 used 10%.
"""

import math


def wsd_lr(step: int,
           total_steps: int,
           peak_lr: float = 5e-4,
           min_lr: float = 5e-5,
           warmup_steps: int = 2000,
           decay_fraction: float = 0.10,
           decay_style: str = "linear") -> float:
    """Learning rate at a given step.

        [0, warmup_steps)            linear ramp 0 -> peak_lr
        [warmup_steps, decay_start)  constant at peak_lr
        [decay_start, total_steps)   peak_lr -> min_lr
        [total_steps, inf)           min_lr
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    decay_steps = int(total_steps * decay_fraction)
    decay_start = total_steps - decay_steps

    if step < decay_start:
        return peak_lr
    if step >= total_steps:
        return min_lr

    progress = (step - decay_start) / max(decay_steps, 1)

    if decay_style == "linear":
        factor = 1.0 - progress
    elif decay_style == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"unknown decay_style: {decay_style}")

    return min_lr + (peak_lr - min_lr) * factor


class WSDScheduler:
    """Sets the LR on an optimizer each step.

    Deliberately not an LRScheduler subclass — state is a single int, which
    makes checkpointing trivial.
    """

    def __init__(self, optimizer, total_steps: int, peak_lr: float = 5e-4,
                 min_lr: float = 5e-5, warmup_steps: int = 2000,
                 decay_fraction: float = 0.10, decay_style: str = "linear"):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_fraction = decay_fraction
        self.decay_style = decay_style
        self.step_count = 0
        self.set_lr(0)

    def get_lr(self, step: int | None = None) -> float:
        step = self.step_count if step is None else step
        return wsd_lr(step, self.total_steps, self.peak_lr, self.min_lr,
                      self.warmup_steps, self.decay_fraction, self.decay_style)

    def set_lr(self, step: int) -> float:
        lr = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self) -> float:
        self.step_count += 1
        return self.set_lr(self.step_count)

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "total_steps": self.total_steps}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]
        self.total_steps = state.get("total_steps", self.total_steps)
        self.set_lr(self.step_count)

    def extend(self, new_total_steps: int) -> None:
        """Push the decay phase further out mid-run.

        This is the reason for choosing WSD over cosine: if eval curves are
        still climbing at the planned stopping point, keep going in the stable
        phase rather than restarting.
        """
        if new_total_steps <= self.step_count:
            raise ValueError("new_total_steps must exceed the current step")
        self.total_steps = new_total_steps
        self.set_lr(self.step_count)