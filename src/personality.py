"""
Agent personality system.

Personality fields control autonomous trading behavior and are fixed at
agent creation time — independent of any market the agent may trade in.

Schema (see FINAL_PHASE_TASKS.md):
  check_interval_mean    - average seconds between market polls (>0)
  check_interval_jitter  - uniform noise on poll interval (>=0)
  edge_threshold         - minimum |belief - price| to consider trading ([0, 1])
  participation_rate     - P(actually trade | threshold met) ([0, 1])
  trade_size_noise       - multiplier noise on optimal trade size ([0, 1])
  signal_sensitivity     - weight on new external signals ([0, 1])
  stubbornness           - damping factor for belief updates ([0, 1])

Usage:
    p = sample_personality()                     # draw from default population
    p = sample_personality(distribution_config)  # custom distribution
    p = Personality.from_dict(some_dict)         # deserialize
    d = p.to_dict()                              # serialize
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class Personality:
    check_interval_mean: float = 2.0
    check_interval_jitter: float = 1.0
    edge_threshold: float = 0.03
    participation_rate: float = 0.80
    trade_size_noise: float = 0.20
    signal_sensitivity: float = 0.50
    stubbornness: float = 0.30
    # fraction of CRRA-optimal trade to execute per cycle, matches round-based
    # trade_fraction=0.20 default that prevents buy/liquidate oscillation
    trade_fraction: float = 0.20
    # weight on the crowd-belief signal sourced from recent comments; gated
    # globally by COMMENTS_INFLUENCE_TRADERS, zero default = no effect
    comment_influence: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialize to plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Personality":
        """Deserialize from dict, falling back to field defaults for missing keys."""
        defaults = cls()
        return cls(
            check_interval_mean=float(d.get("check_interval_mean", defaults.check_interval_mean)),
            check_interval_jitter=float(d.get("check_interval_jitter", defaults.check_interval_jitter)),
            edge_threshold=float(d.get("edge_threshold", defaults.edge_threshold)),
            participation_rate=float(d.get("participation_rate", defaults.participation_rate)),
            trade_size_noise=float(d.get("trade_size_noise", defaults.trade_size_noise)),
            signal_sensitivity=float(d.get("signal_sensitivity", defaults.signal_sensitivity)),
            stubbornness=float(d.get("stubbornness", defaults.stubbornness)),
            trade_fraction=float(d.get("trade_fraction", defaults.trade_fraction)),
            comment_influence=float(d.get("comment_influence", defaults.comment_influence)),
        )


# Default population distribution used when no explicit distribution is configured.
# Fields sampled from Uniform distributions per the spec; others are fixed at defaults.
DEFAULT_POPULATION_DIST: Dict[str, Any] = {
    "check_interval_mean": {"dist": "uniform", "low": 1.0, "high": 4.0},
    "check_interval_jitter": {"dist": "fixed", "value": 1.0},
    "edge_threshold": {"dist": "uniform", "low": 0.01, "high": 0.10},
    "participation_rate": {"dist": "uniform", "low": 0.50, "high": 1.0},
    "trade_size_noise": {"dist": "fixed", "value": 0.20},
    "signal_sensitivity": {"dist": "fixed", "value": 0.50},
    "stubbornness": {"dist": "fixed", "value": 0.30},
    "trade_fraction": {"dist": "fixed", "value": 0.20},
    "comment_influence": {"dist": "uniform", "low": 0.0, "high": 0.3},
}


def sample_personality(
    distribution_config: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
) -> Personality:
    """
    Draw a Personality from a distribution config.

    Each field entry in *distribution_config* must be one of:
        {"dist": "uniform", "low": L, "high": H}
        {"dist": "normal",  "mean": M, "std": S}
        {"dist": "fixed",   "value": V}

    Missing field entries fall back to the ``Personality`` dataclass defaults.

    Args:
        distribution_config: per-field distribution specs; defaults to
            ``DEFAULT_POPULATION_DIST`` when omitted.
        rng: seeded :class:`random.Random` instance for reproducibility;
            a fresh unseeded instance is used when omitted.

    Returns:
        A new :class:`Personality` sampled from the given distributions.
    """
    if distribution_config is None:
        distribution_config = DEFAULT_POPULATION_DIST
    if rng is None:
        rng = random.Random()

    defaults = Personality()
    fields: Dict[str, float] = {}

    for field_name in Personality.__dataclass_fields__:
        cfg = distribution_config.get(field_name, {})
        dist = cfg.get("dist", "fixed")
        default_val = float(getattr(defaults, field_name))

        if dist == "uniform":
            val = rng.uniform(float(cfg["low"]), float(cfg["high"]))
        elif dist == "normal":
            val = rng.gauss(float(cfg["mean"]), float(cfg["std"]))
        else:  # fixed or unrecognised
            val = float(cfg.get("value", default_val))

        fields[field_name] = val

    return Personality(**fields)
