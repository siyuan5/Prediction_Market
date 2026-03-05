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


def test_trade_fraction_scales_volume():
    # a lower trade_fraction should produce strictly less total volume
    common = dict(
        seed=7,
        ground_truth=0.70,
        n_agents=20,
        n_rounds=20,
        b=500.0,
        signal_spec=SignalSpec(mode="binomial", n=25),
    )
    r_full = run_phase2(**common, trade_fraction=1.0)
    r_half = run_phase2(**common, trade_fraction=0.5)

    vol_full = sum(r_full["trade_volume"])
    vol_half = sum(r_half["trade_volume"])
    assert vol_half < vol_full


def test_price_converges_to_ground_truth():
    # with stale-price fix and trade dampening, final price should land near p*
    r = run_phase2(
        seed=42,
        ground_truth=0.70,
        n_agents=50,
        n_rounds=100,
        initial_cash=100.0,
        sigma=0.10,
        rho_values=[0.75, 1.0, 1.25, 1.5, 2.0],
        b=500.0,
        signal_spec=SignalSpec(mode="binomial", n=25),
        belief_update_method="beta",
        prior_strength=20.0,
        obs_strength=10.0,
        trade_fraction=0.20,
        shuffle_agents=False,
    )
    assert r["final_error"] < 0.05


def test_error_trends_downward():
    # average error in the second half of the run should be less than the first half
    r = run_phase2(
        seed=0,
        ground_truth=0.65,
        n_agents=40,
        n_rounds=80,
        initial_cash=100.0,
        sigma=0.10,
        b=500.0,
        signal_spec=SignalSpec(mode="binomial", n=25),
        belief_update_method="beta",
        prior_strength=20.0,
        obs_strength=10.0,
        trade_fraction=0.20,
        shuffle_agents=False,
    )
    errors = r["error_series"]
    mid = len(errors) // 2
    early_avg = sum(errors[:mid]) / mid
    late_avg = sum(errors[mid:]) / (len(errors) - mid)
    assert late_avg < early_avg