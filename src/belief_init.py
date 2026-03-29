# Agent belief initialization: BeliefSpec configures agent priors for simulation runs.
# This design mirrors SignalSpec in phase2_utils.py for consistency.
# Pass a BeliefSpec to SimulationEngine or call sample_beliefs() for standalone usage.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class BeliefSpec:
    """
    Describes how to initialize agent beliefs at the start of a simulation.

    mode options:
      - "gaussian": Beliefs ~ N(ground_truth, sigma), clipped to [low, high]
      - "uniform":  Random beliefs uniformly in [low, high], ignores ground_truth
      - "bimodal":  Population split across specified centers (optionally weighted), each with jitter
      - "fixed":    All agents share the same belief
    """
    mode: str = "gaussian"

    # Used for "gaussian" mode (beliefs centered around ground_truth)
    sigma: float = 0.10

    # Used for "bimodal" mode
    group_beliefs: List[float] = field(default_factory=lambda: [0.3, 0.9])  # cluster means
    group_weights: Optional[List[float]] = None  # probability for each group; uniform if None
    group_noise: float = 0.02  # per-agent jitter around cluster means

    # Used for "fixed" mode
    fixed_value: float = 0.70


def sample_beliefs(
    ground_truth: float,
    n_agents: int,
    spec: BeliefSpec,
    rng: np.random.Generator,
    low: float = 0.01,
    high: float = 0.99,
) -> np.ndarray:
    """
    Generate initial agent beliefs with shape (n_agents,), according to BeliefSpec.
    All outputs are clipped to [low, high].
    """
    if spec.mode == "gaussian":
        # Normal dist around ground_truth with specified sigma
        samples = rng.normal(loc=ground_truth, scale=spec.sigma, size=n_agents)

    elif spec.mode == "uniform":
        # Uniform random in [low, high], ignores ground_truth
        samples = rng.uniform(low=low, high=high, size=n_agents)

    elif spec.mode == "bimodal":
        # Draw agents from clusters with centers in group_beliefs.
        n_groups = len(spec.group_beliefs)
        weights = spec.group_weights
        if weights is None:
            # Even split if not specified
            weights = [1.0 / n_groups] * n_groups
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        # Assign each agent to a group according to weights
        group_ids = rng.choice(n_groups, size=n_agents, p=weights)
        centers = np.array(spec.group_beliefs, dtype=float)
        samples = centers[group_ids].copy()
        if spec.group_noise > 0:
            # Add per-agent Gaussian noise if group_noise > 0
            samples += rng.normal(0.0, spec.group_noise, size=n_agents)

    elif spec.mode == "fixed":
        # Everyone uses the same belief value
        samples = np.full(n_agents, float(spec.fixed_value))

    else:
        raise ValueError(
            f"unknown BeliefSpec.mode={spec.mode!r}, "
            "pick from: 'gaussian', 'uniform', 'bimodal', 'fixed'"
        )

    return np.clip(samples, low, high)
