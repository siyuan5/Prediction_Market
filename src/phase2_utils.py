# phase2_utils.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def clip_prob(p: float, low: float = 0.01, high: float = 0.99) -> float:
    return float(np.clip(p, low, high))


@dataclass(frozen=True)
class SignalSpec:
    """
    How we generate the public signal S_t each round.

    mode:
      - "bernoulli": S_t in {0,1} with P(S_t=1)=P*
      - "binomial": S_t = k/n where k ~ Binomial(n, P*)
      - "gaussian": S_t ~ Normal(P*, sigma) then clipped to [0.01, 0.99]
    """
    mode: str = "binomial"
    n: int = 25          # for binomial
    sigma: float = 0.08  # for gaussian
    low: float = 0.01
    high: float = 0.99


def generate_signal(ground_truth: float, rng: np.random.Generator, spec: SignalSpec) -> float:
    pstar = float(ground_truth)
    if spec.mode == "bernoulli":
        return float(rng.random() < pstar)
    if spec.mode == "binomial":
        n = max(int(spec.n), 1)
        k = rng.binomial(n=n, p=pstar)
        return float(k / n)
    if spec.mode == "gaussian":
        s = rng.normal(loc=pstar, scale=float(spec.sigma))
        return clip_prob(float(s), spec.low, spec.high)
    raise ValueError(f"Unknown SignalSpec.mode={spec.mode!r}")


def update_belief_weighted(prior_p: float, signal_s: float, w: float) -> float:
    """
    Simple Phase 2 update:
        p_new = (1-w)*p_old + w*S_t
    """
    w = float(w)
    if not (0.0 <= w <= 1.0):
        raise ValueError("w must be in [0,1]")
    p_new = (1.0 - w) * float(prior_p) + w * float(signal_s)
    return clip_prob(p_new)


def update_belief_beta(prior_p: float, signal_s: float, prior_strength: float, obs_strength: float) -> float:
    """
    Beta-Binomial flavored update using pseudo-counts.

    Interpret prior as Beta(alpha0, beta0) with:
        alpha0 = prior_p * prior_strength
        beta0  = (1-prior_p) * prior_strength

    Interpret "signal" S_t in [0,1] as k/n with n=obs_strength (pseudo),
        alpha1 = alpha0 + k
        beta1  = beta0 + (n-k)
        posterior mean = alpha1 / (alpha1+beta1)
    """
    prior_p = clip_prob(float(prior_p))
    s = clip_prob(float(signal_s))

    prior_strength = float(prior_strength)
    obs_strength = float(obs_strength)
    if prior_strength <= 0 or obs_strength <= 0:
        raise ValueError("prior_strength and obs_strength must be > 0")

    alpha0 = prior_p * prior_strength
    beta0 = (1.0 - prior_p) * prior_strength

    n = obs_strength
    k = s * n

    alpha1 = alpha0 + k
    beta1 = beta0 + (n - k)

    post_mean = alpha1 / (alpha1 + beta1)
    return clip_prob(float(post_mean))
