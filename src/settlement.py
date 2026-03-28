"""
Terminal resolution and P&L for binary Yes shares.

Draws Yes/No from P* = ground_truth, then terminal_wealth = cash + shares * payoff
(payoff 1 if Yes, 0 if No). Used by the API after SimulationEngine runs.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def compute_settlement(
    agents_final: List[Dict[str, Any]],
    *,
    initial_cash: float,
    ground_truth: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Binary resolution: draw Yes with probability P* = ground_truth (reproducible RNG).
    Each agent holds net Yes-contract shares z: pays z * 1 if Yes, z * 0 if No (short Yes if z < 0).
    Terminal wealth = cash + z * payoff; profit vs start = terminal - initial_cash.
    """
    rng = np.random.default_rng(int(seed) + 1_403_817_293)
    u = float(rng.random())
    outcome_yes = bool(u < float(ground_truth))
    pay = 1.0 if outcome_yes else 0.0
    ic = float(initial_cash)

    scored: List[Dict[str, Any]] = []
    for a in agents_final:
        cash = float(a["cash"])
        shares = float(a["shares"])
        terminal_wealth = cash + shares * pay
        profit = terminal_wealth - ic
        scored.append(
            {
                "agent_id": int(a["agent_id"]),
                "belief": float(a["belief"]),
                "rho": float(a["rho"]),
                "cash_pre_settlement": cash,
                "yes_shares": shares,
                "terminal_wealth": terminal_wealth,
                "profit": profit,
            }
        )

    by_profit_desc = sorted(scored, key=lambda r: r["profit"], reverse=True)
    by_profit_asc = sorted(scored, key=lambda r: r["profit"])

    return {
        "outcome": "yes" if outcome_yes else "no",
        "outcome_is_yes": outcome_yes,
        "resolution_draw_u": u,
        "p_star": float(ground_truth),
        "initial_cash": ic,
        "payoff_per_yes_share": pay,
        "winners": by_profit_desc[:10],
        "losers": by_profit_asc[:10],
    }
