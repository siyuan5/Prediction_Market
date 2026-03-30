"""Smoke tests for `src.simulation_engine.SimulationEngine` (no production code changes)."""

from __future__ import annotations

import math

from belief_init import BeliefSpec
from phase2_utils import SignalSpec
from simulation_engine import SimulationEngine


def test_lmsr_phase1_run_advances_round_and_price_in_unit_interval():
    eng = SimulationEngine(
        mechanism="lmsr",
        phase=1,
        seed=11,
        ground_truth=0.65,
        n_agents=12,
        initial_cash=100.0,
        b=200.0,
        belief_spec=BeliefSpec(mode="gaussian", sigma=0.08),
        shuffle_agents=False,
    )
    assert eng.round == 0
    out = eng.run(5)
    assert eng.round == 5
    assert out["rounds_run"] == 5
    assert len(eng.price_series) == 5
    assert all(0.0 < p < 1.0 for p in eng.price_series)
    st = eng.get_state()
    assert st["round"] == 5
    assert math.isclose(st["error"], abs(st["price"] - 0.65))


def test_lmsr_phase2_emits_one_signal_per_round():
    n = 8
    eng = SimulationEngine(
        mechanism="lmsr",
        phase=2,
        seed=3,
        ground_truth=0.70,
        n_agents=15,
        b=150.0,
        signal_spec=SignalSpec(mode="binomial", n=20),
        shuffle_agents=False,
    )
    eng.run(n)
    assert len(eng.signal_series) == n
    assert len(eng.mean_belief_series) == n


def test_shift_beliefs_appends_event_and_changes_targets():
    eng = SimulationEngine(
        mechanism="lmsr",
        phase=1,
        seed=0,
        n_agents=6,
        b=100.0,
        shuffle_agents=False,
    )
    eng.run(1)
    ev = eng.shift_beliefs(new_belief=0.55)
    assert ev["n_agents_shifted"] == 6
    assert math.isclose(ev["after_mean"], 0.55)
    assert len(eng.belief_shift_events) == 1
    assert all(abs(a.belief - 0.55) < 1e-9 for a in eng.agents)


def test_cda_smoke_bounded_prices_and_volume_nonnegative():
    eng = SimulationEngine(
        mechanism="cda",
        phase=1,
        seed=21,
        n_agents=10,
        initial_cash=100.0,
        b=100.0,
        shuffle_agents=False,
    )
    eng.run(6)
    assert all(0.01 <= p <= 0.99 for p in eng.price_series)
    assert all(v >= 0.0 for v in eng.trade_volume)
    assert len(eng.best_bid_series) == 6
    assert len(eng.best_ask_series) == 6


def test_invalid_mechanism_raises():
    try:
        SimulationEngine(mechanism="auction", phase=1, seed=1, n_agents=4)
    except ValueError as e:
        assert "mechanism" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for invalid mechanism")


def test_get_agents_pnl_matches_mark_definition():
    eng = SimulationEngine(
        mechanism="lmsr",
        phase=1,
        seed=5,
        n_agents=4,
        b=300.0,
        shuffle_agents=False,
    )
    eng.run(3)
    price = eng.get_state()["price"]
    for row in eng.get_agents():
        pnl = row["pnl"]
        expected = row["cash"] + row["shares"] * price - eng.initial_cash
        assert math.isclose(pnl, expected, rel_tol=0, abs_tol=1e-6)
