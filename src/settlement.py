"""
Terminal resolution and P&L calculation for binary (Yes/No) market shares.

- This function draws the market outcome ('yes' or 'no') randomly, with P(yes) = ground_truth.
- After the outcome, agent wealth is updated: terminal_wealth = cash + shares * payoff (payoff = 1 for yes, 0 for no).
- Profit is terminal_wealth minus initial_cash, for easy ranking of agent performance.
- Used by the API after SimulationEngine runs.
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
    Resolve market, compute post-settlement wealth and profit for each agent.

    - Randomly draw market outcome (Yes/No) using ground_truth probability, with deterministic seed.
    - Compute for each agent:
        - Terminal wealth: cash + yes_shares * payoff
        - Profit: terminal_wealth - initial_cash
    - Return top (winners) and bottom (losers) agents by profit, plus outcome details.
    """
    # Seed offsets avoid collisions with other RNG streams in the repo
    rng = np.random.default_rng(int(seed) + 1_403_817_293)
    u = float(rng.random())  # [0, 1) uniform draw for outcome resolution
    outcome_yes = bool(u < float(ground_truth))  # Market outcome: True = Yes, False = No
    pay = 1.0 if outcome_yes else 0.0  # Per share payoff: 1 for Yes, 0 for No
    ic = float(initial_cash)

    scored: List[Dict[str, Any]] = []
    for a in agents_final:
        # Each agent's pre-settlement state
        cash = float(a["cash"])
        shares = float(a["shares"])
        terminal_wealth = cash + shares * pay  # Realized after settlement
        profit = terminal_wealth - ic  # P&L relative to initial endowment
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

    # Rank agents by final profit
    by_profit_desc = sorted(scored, key=lambda r: r["profit"], reverse=True)
    by_profit_asc = sorted(scored, key=lambda r: r["profit"])

    return {
        "outcome": "yes" if outcome_yes else "no",
        "outcome_is_yes": outcome_yes,
        "resolution_draw_u": u,
        "p_star": float(ground_truth),
        "initial_cash": ic,
        "payoff_per_yes_share": pay,
        "winners": by_profit_desc[:10],  # Top 10 agents by profit
        "losers": by_profit_asc[:10],    # Bottom 10 agents by profit
    }
