import math

from phase2_utils import SignalSpec
from team_a_phase2_simulation import run_phase2


def test_seed_reproducibility():
    params = dict(
        seed=123,
        ground_truth=0.70,
        n_agents=30,
        n_rounds=40,
        initial_cash=100.0,
        sigma=0.10,
        rho_values=[0.75, 1.0, 1.25],
        b=2000.0,
        signal_spec=SignalSpec(mode="binomial", n=25),
        belief_update_method="beta",
        prior_strength=20.0,
        obs_strength=5.0,
        shuffle_agents=False,
    )

    r1 = run_phase2(**params)
    r2 = run_phase2(**params)

    assert r1["signal_series"] == r2["signal_series"]
    assert r1["price_series"] == r2["price_series"]
    assert r1["mean_belief_series"] == r2["mean_belief_series"]
    assert r1["final_price"] == r2["final_price"]
    assert r1["final_error"] == r2["final_error"]


def test_price_tracking_shapes_and_bounds():
    r = run_phase2(
        seed=7,
        ground_truth=0.65,
        n_agents=20,
        n_rounds=25,
        b=1000.0,
        signal_spec=SignalSpec(mode="binomial", n=25),
        shuffle_agents=False,
    )

    assert len(r["signal_series"]) == 25
    assert len(r["price_series"]) == 25
    assert len(r["mean_belief_series"]) == 25
    assert len(r["error_series"]) == 25

    # price should be a probability
    assert all(0.0 <= q <= 1.0 for q in r["price_series"])

    # final_price should match last price in the series
    assert math.isclose(r["final_price"], r["price_series"][-1], rel_tol=0, abs_tol=1e-12)

    # error_series should equal |price - P*| each round
    gt = r["ground_truth"]
    for q, e in zip(r["price_series"], r["error_series"]):
        assert math.isclose(e, abs(q - gt), rel_tol=0, abs_tol=1e-12)


def test_belief_updates_change_mean_belief():
    r = run_phase2(
        seed=99,
        ground_truth=0.80,
        n_agents=50,
        n_rounds=60,
        sigma=0.20,
        signal_spec=SignalSpec(mode="binomial", n=50),
        belief_update_method="beta",
        prior_strength=10.0,
        obs_strength=20.0,
        shuffle_agents=False,
    )

    beliefs = r["mean_belief_series"]

    # Beliefs should not be constant (otherwise belief updates aren't applied)
    assert len(set(beliefs)) > 1

    # And they should deviate from the initial mean at least a little
    initial = r["mean_initial_belief"]
    max_dev = max(abs(b - initial) for b in beliefs)
    assert max_dev > 1e-6