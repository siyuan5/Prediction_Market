"""
Tests for ``settlement.compute_settlement``.

Settlement draws a single uniform ``u`` from a seed-derived RNG, resolves YES if
``u < ground_truth``, then values each agent at ``cash + yes_shares * payoff``.
These tests lock in reproducibility, the accounting identities, and ranking logic
without touching application code.
"""

from __future__ import annotations

import numpy as np

from settlement import compute_settlement


def _resolution_u(seed: int) -> float:
    """
    Replicate the first draw used inside ``compute_settlement``.

    Must use the same seed offset as ``settlement.py`` so tests can predict
    ``resolution_draw_u`` and the Yes/No outcome without duplicating implementation
    details beyond the RNG line.
    """
    rng = np.random.default_rng(int(seed) + 1_403_817_293)
    return float(rng.random())


def test_settlement_deterministic_for_same_inputs():
    """
    Pure function behavior: identical inputs must yield identical outputs.

    If this fails, something in settlement became non-deterministic (e.g. dict
    ordering assumptions, or RNG usage drifted).
    """
    agents = [
        {"agent_id": 0, "belief": 0.5, "rho": 1.0, "cash": 80.0, "shares": 20.0},
        {"agent_id": 1, "belief": 0.6, "rho": 1.0, "cash": 100.0, "shares": 0.0},
    ]
    a = compute_settlement(agents, initial_cash=100.0, ground_truth=0.72, seed=999)
    b = compute_settlement(agents, initial_cash=100.0, ground_truth=0.72, seed=999)
    assert a == b


def test_settlement_wealth_and_profit_match_outcome():
    """
    End-to-end check for one agent: outcome bit, reported draw, and P&L.

    We recompute ``u`` and YES/NO outside the module, then assert the returned
    payload matches. Terminal wealth must be ``cash + shares * pay`` with
    ``pay ∈ {0,1}``; profit is that wealth minus the common ``initial_cash`` endowment.
    """
    seed = 42
    ground_truth = 0.65
    initial_cash = 100.0
    u = _resolution_u(seed)
    outcome_yes = u < ground_truth
    pay = 1.0 if outcome_yes else 0.0

    agents = [
        {"agent_id": 0, "belief": 0.4, "rho": 1.0, "cash": 50.0, "shares": 50.0},
    ]
    r = compute_settlement(agents, initial_cash=initial_cash, ground_truth=ground_truth, seed=seed)

    # Exposed resolution state should match our independent derivation
    assert r["resolution_draw_u"] == u
    assert r["outcome_is_yes"] == outcome_yes
    assert r["payoff_per_yes_share"] == pay
    assert r["p_star"] == ground_truth
    assert r["initial_cash"] == initial_cash

    tw = 50.0 + 50.0 * pay
    assert r["winners"][0]["terminal_wealth"] == tw
    assert r["winners"][0]["profit"] == tw - initial_cash


def test_settlement_winners_sorted_by_profit():
    """
    Ranking invariants for the ``winners`` / ``losers`` slices (top/bottom 10).

    Every row’s ``profit`` must match ``cash + shares*payoff - initial_cash``.
    ``winners`` must be descending by profit; ``losers`` ascending (worst first).
    With two agents both lists contain the same two rows, ordered oppositely.
    """
    agents = [
        {"agent_id": 0, "belief": 0.5, "rho": 1.0, "cash": 120.0, "shares": 0.0},
        {"agent_id": 1, "belief": 0.5, "rho": 1.0, "cash": 80.0, "shares": 40.0},
    ]
    r = compute_settlement(agents, initial_cash=100.0, ground_truth=0.5, seed=7)
    pay = r["payoff_per_yes_share"]
    expected_profit = {
        0: float(agents[0]["cash"]) + float(agents[0]["shares"]) * pay - 100.0,
        1: float(agents[1]["cash"]) + float(agents[1]["shares"]) * pay - 100.0,
    }
    for row in r["winners"]:
        assert row["profit"] == expected_profit[row["agent_id"]]
    win_profits = [row["profit"] for row in r["winners"]]
    assert win_profits == sorted(win_profits, reverse=True)
    lose_profits = [row["profit"] for row in r["losers"]]
    assert lose_profits == sorted(lose_profits)
