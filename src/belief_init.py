# configurable agent belief initialization
# mirrors the SignalSpec pattern from phase2_utils.py
# pass a BeliefSpec to SimulationEngine or call sample_beliefs() directly

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class BeliefSpec:
    # how beliefs are distributed at the start of a run
    #
    # modes:
    #   "gaussian" - default, same as before (clipped normal around ground truth)
    #   "uniform"  - random spread across [0.01, 0.99], ignores ground truth
    #   "bimodal"  - agents split into groups at different belief centers
    #                e.g. half at 0.3, half at 0.9 (polarized market)
    #   "fixed"    - everyone starts with the same belief

    mode: str = "gaussian"

    # gaussian
    sigma: float = 0.10

    # bimodal
    group_beliefs: List[float] = field(default_factory=lambda: [0.3, 0.9])
    group_weights: Optional[List[float]] = None  # none = equal split
    group_noise: float = 0.02                    # small jitter per agent

    # fixed
    fixed_value: float = 0.70


def sample_beliefs(
    ground_truth: float,
    n_agents: int,
    spec: BeliefSpec,
    rng: np.random.Generator,
    low: float = 0.01,
    high: float = 0.99,
) -> np.ndarray:
    # returns shape (n_agents,) clipped to [low, high]
    if spec.mode == "gaussian":
        samples = rng.normal(loc=ground_truth, scale=spec.sigma, size=n_agents)

    elif spec.mode == "uniform":
        samples = rng.uniform(low=low, high=high, size=n_agents)

    elif spec.mode == "bimodal":
        n_groups = len(spec.group_beliefs)
        weights = spec.group_weights
        if weights is None:
            weights = [1.0 / n_groups] * n_groups
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        group_ids = rng.choice(n_groups, size=n_agents, p=weights)
        centers = np.array(spec.group_beliefs, dtype=float)
        samples = centers[group_ids].copy()
        if spec.group_noise > 0:
            samples += rng.normal(0.0, spec.group_noise, size=n_agents)

    elif spec.mode == "fixed":
        samples = np.full(n_agents, float(spec.fixed_value))

    else:
        raise ValueError(
            f"unknown BeliefSpec.mode={spec.mode!r}, "
            "pick from: 'gaussian', 'uniform', 'bimodal', 'fixed'"
        )

    return np.clip(samples, low, high)
