"""Tests for `src.settlement.compute_settlement` (deterministic RNG + P&L math)."""

from __future__ import annotations

import numpy as np

from settlement import compute_settlement


def _resolution_u(seed: int) -> float:
    """Mirror settlement's RNG stream for the outcome draw."""
    rng = np.random.default_rng(int(seed) + 1_403_817_293)
    return float(rng.random())


def test_settlement_deterministic_for_same_inputs():
    agents = [
        {"agent_id": 0, "belief": 0.5, "rho": 1.0, "cash": 80.0, "shares": 20.0},
        {"agent_id": 1, "belief": 0.6, "rho": 1.0, "cash": 100.0, "shares": 0.0},
    ]
    a = compute_settlement(agents, initial_cash=100.0, ground_truth=0.72, seed=999)
    b = compute_settlement(agents, initial_cash=100.0, ground_truth=0.72, seed=999)
    assert a == b


def test_settlement_wealth_and_profit_match_outcome():
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

    assert r["resolution_draw_u"] == u
    assert r["outcome_is_yes"] == outcome_yes
    assert r["payoff_per_yes_share"] == pay
    assert r["p_star"] == ground_truth
    assert r["initial_cash"] == initial_cash

    tw = 50.0 + 50.0 * pay
    assert r["winners"][0]["terminal_wealth"] == tw
    assert r["winners"][0]["profit"] == tw - initial_cash


def test_settlement_winners_sorted_by_profit():
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
