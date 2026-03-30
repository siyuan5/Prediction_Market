"""
Smoke and contract tests for ``simulation_engine.SimulationEngine``.

The engine is the shared driver for CLI scripts, FastAPI, and the UI: it wires
agents to either LMSR or CDA, records time series, and supports mid-run belief
shocks. These tests use small agent counts and fixed seeds so runs are fast and
stable; ``shuffle_agents=False`` removes ordering randomness where we only care
about aggregates.
"""

from __future__ import annotations

import math

from belief_init import BeliefSpec
from phase2_utils import SignalSpec
from simulation_engine import SimulationEngine


def test_lmsr_phase1_run_advances_round_and_price_in_unit_interval():
    """
    LMSR phase 1: beliefs are fixed; each round only trades.

    After ``run(5)``, internal ``round`` and ``price_series`` length must match,
    LMSR prices stay strictly inside (0,1), and ``get_state()`` error is the
    absolute gap between price and ground truth (API contract for charts).
    """
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
    """
    Phase 2: before trading each round, every agent sees one public signal.

    Therefore ``signal_series`` should gain exactly one entry per round, and the
    parallel ``mean_belief_series`` should stay the same length (one aggregate
    belief snapshot after updates).
    """
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
    """
    Mid-run belief shock API: ``shift_beliefs(new_belief=...)`` clips to (0.01,0.99).

    We run one round first so ``round`` is non-zero, then shift everyone to 0.55.
    The returned event should report six agents shifted and mean 0.55; the log
    ``belief_shift_events`` gains one entry; all agents match the new belief.
    """
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
    """
    CDA path: ``ContinuousDoubleAuction`` reference prices and order-book stats.

    Reference prices stay in the model’s traded range; volume is nonnegative;
    best bid/ask history exists for each round (UI order-book strip).
    """
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
    """
    Constructor validation: only ``lmsr`` and ``cda`` are supported.

    A bad string should fail fast with ``ValueError`` so misconfigured API
    requests or scripts do not silently fall through.
    """
    try:
        SimulationEngine(mechanism="auction", phase=1, seed=1, n_agents=4)
    except ValueError as e:
        assert "mechanism" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for invalid mechanism")


def test_get_agents_pnl_matches_mark_definition():
    """
    ``get_agents`` P&L matches mark-to-market at the current LMSR price.

    For each agent, ``pnl`` should equal ``cash + shares * price - initial_cash``
    where ``price`` is the engine’s current quote (same as ``get_state()['price']``).
    This mirrors how the API exposes agent rows to the frontend.
    """
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
