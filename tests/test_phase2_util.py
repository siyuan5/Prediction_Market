import numpy as np

from phase2_utils import (
    SignalSpec,
    generate_signal,
    update_belief_weighted,
    update_belief_beta,
)


def test_generate_signal_binomial_mean_close_to_ground_truth():
    rng = np.random.default_rng(0)
    pstar = 0.70
    spec = SignalSpec(mode="binomial", n=50)
    samples = [generate_signal(pstar, rng, spec) for _ in range(2000)]
    m = float(np.mean(samples))
    assert abs(m - pstar) < 0.03


def test_weighted_update_moves_toward_signal():
    p0 = 0.30
    s = 0.80
    p1 = update_belief_weighted(p0, s, w=0.25)
    assert p0 < p1 < s


def test_beta_update_moves_toward_signal_and_clips():
    p0 = 0.98
    s = 0.05
    p1 = update_belief_beta(p0, s, prior_strength=20.0, obs_strength=10.0)
    assert 0.01 <= p1 <= 0.99
    assert p1 < p0
