# Signal and update utility functions for Phase 2 (public signals, agent Bayesian update).

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def clip_prob(p: float, low: float = 0.01, high: float = 0.99) -> float:
    """Clip probability p to interval [low, high]."""
    return float(np.clip(p, low, high))


@dataclass(frozen=True)
class SignalSpec:
    """
    Configuration for generating public signal S_t in each round.

    mode options:
      - "bernoulli":    S_t ∈ {0,1}, S_t=1 with probability P*
      - "binomial":     S_t = k/n, with k ~ Binomial(n, P*) (adds signal reliability)
      - "gaussian":     S_t ~ N(P*, sigma), clipped to [low, high]
    """
    mode: str = "binomial"
    n: int = 25          # Number of samples for binomial mode
    sigma: float = 0.08  # Stddev for gaussian mode
    low: float = 0.01    # Minimum possible signal value
    high: float = 0.99   # Maximum possible signal value

def generate_signal(ground_truth: float, rng: np.random.Generator, spec: SignalSpec) -> float:
    """
    Draw a public signal S_t based on ground truth P* and the provided SignalSpec.
    Returns S_t in [0, 1] for binomial/gaussian, or {0,1} for bernoulli.
    """
    pstar = float(ground_truth)
    if spec.mode == "bernoulli":
        # Simple: S_t=1 with probability P*, else 0
        return float(rng.random() < pstar)
    if spec.mode == "binomial":
        # Binomial aggregation: average of n noisy draws
        n = max(int(spec.n), 1)
        k = rng.binomial(n=n, p=pstar)
        return float(k / n)
    if spec.mode == "gaussian":
        # Gaussian signal, clipped for numerical robustness
        s = rng.normal(loc=pstar, scale=float(spec.sigma))
        return clip_prob(float(s), spec.low, spec.high)
    # Catch invalid mode for explicit failure
    raise ValueError(f"Unknown SignalSpec.mode={spec.mode!r}")

def update_belief_weighted(prior_p: float, signal_s: float, w: float) -> float:
    """
    Weighted update (convex combination) based on signal_s and prior_p:
        p_new = (1-w) * prior_p + w * signal_s
    Used for simple intuition or as a baseline Bayesian-like filter.
    """
    w = float(w)
    if not (0.0 <= w <= 1.0):
        raise ValueError("w must be in [0,1]")
    p_new = (1.0 - w) * float(prior_p) + w * float(signal_s)
    return clip_prob(p_new)

def update_belief_beta(prior_p: float, signal_s: float, prior_strength: float, obs_strength: float) -> float:
    """
    Bayesian update using Beta pseudo-counts for each round:
      - 'prior_p': agent's previous belief
      - 'signal_s': observed signal in [0,1]
      - 'prior_strength': how much to trust the prior (as pseudo-observations)
      - 'obs_strength': how much to trust the new signal (as pseudo-observations)

    Interpretation: prior ~ Beta(alpha0, beta0)
      alpha0 = prior_p * prior_strength
      beta0  = (1-prior_p) * prior_strength
    Treat signal_s as k/n with n=obs_strength pseudo-counts:
      alpha1 = alpha0 + k
      beta1  = beta0 + (n-k)
      posterior mean returned as new belief.
    """
    prior_p = clip_prob(float(prior_p))
    s = clip_prob(float(signal_s))

    prior_strength = float(prior_strength)
    obs_strength = float(obs_strength)
    if prior_strength <= 0 or obs_strength <= 0:
        raise ValueError("prior_strength and obs_strength must be > 0")

    alpha0 = prior_p * prior_strength
    beta0 = (1.0 - prior_p) * prior_strength

    # Interpret the "signal" as k successes in n pseudo-trials
    n = obs_strength
    k = s * n

    alpha1 = alpha0 + k
    beta1 = beta0 + (n - k)

    post_mean = alpha1 / (alpha1 + beta1)
    return clip_prob(float(post_mean))
