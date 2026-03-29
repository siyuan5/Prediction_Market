import numpy as np

from phase2_utils import (
    SignalSpec,
    generate_signal,
    update_belief_weighted,
    update_belief_beta,
)

# Ensure binomial signal generator's mean approximates ground truth (law of large numbers)
def test_generate_signal_binomial_mean_close_to_ground_truth():
    rng = np.random.default_rng(0)
    pstar = 0.70
    spec = SignalSpec(mode="binomial", n=50)
    samples = [generate_signal(pstar, rng, spec) for _ in range(2000)]
    m = float(np.mean(samples))
    # Check that the average signal is close to the true probability
    assert abs(m - pstar) < 0.03

# Weighted update: belief moves toward observed signal (not trivial if w < 1)
def test_weighted_update_moves_toward_signal():
    p0 = 0.30
    s = 0.80
    p1 = update_belief_weighted(p0, s, w=0.25)
    # New belief should be strictly between old belief and observed signal
    assert p0 < p1 < s

# Beta update: belief toward signal and result stays within bounds [0.01, 0.99]
def test_beta_update_moves_toward_signal_and_clips():
    p0 = 0.98
    s = 0.05
    # Large prior (20) and moderately large obs (10) should pull belief noticeably but not enough to leave [0.01, 0.99]
    p1 = update_belief_beta(p0, s, prior_strength=20.0, obs_strength=10.0)
    # Ensure clipping to avoid numerically degenerate beliefs
    assert 0.01 <= p1 <= 0.99
    # Belief should be reduced toward s (since s << p0)
    assert p1 < p0
