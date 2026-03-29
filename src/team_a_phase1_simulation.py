import numpy as np

from crra_agent import CRRAAgent
from team_a_market_logic import LMSRMarketMaker


def clipped_gaussian(mean, sigma, size, low=0.01, high=0.99, rng=None):
    """
    Draw normal samples centered at mean with std `sigma`, then clip to (low, high).
    Used to initialize agent beliefs near ground truth, avoiding degenerate 0/1 values.
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(loc=mean, scale=sigma, size=size)
    return np.clip(samples, low, high)


def run_phase1(
    *,
    seed=42,
    ground_truth=0.70,
    n_agents=50,
    n_rounds=100,
    initial_cash=100.0,
    sigma=0.10,
    rho_values=None,
    b=100.0,
):
    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(seed)

    if rho_values is None:
        # Default: moderate mix of risk aversion levels
        rho_values = [0.5, 1.0, 2.0]

    # Assign initial beliefs (static in phase 1) and risk avership
    beliefs = clipped_gaussian(ground_truth, sigma, n_agents, rng=rng)
    rhos = rng.choice(rho_values, size=n_agents, replace=True)

    # Instantiate all agents with cash, belief, and risk preference
    agents = [
        CRRAAgent(
            agent_id=i,
            initial_cash=initial_cash,
            belief_p=float(beliefs[i]),
            rho=float(rhos[i]),
        )
        for i in range(n_agents)
    ]

    # Initialize LMSR market maker with parameter b ("liquidity")
    market = LMSRMarketMaker(b=b)

    price_series = []    # Tracks price at each round
    trade_volume = []    # Tracks total volume per round

    # Main trading loop (no information updates in phase 1)
    for _ in range(n_rounds):
        q_t = market.get_price()  # Observe price before trades for the round
        price_series.append(q_t)

        round_volume = 0.0
        for agent in agents:
            x_star = agent.get_optimal_trade(q_t)  # Agent's optimal trade size (could be 0)
            if abs(x_star) < 1e-9:  # Skip if trade is negligible
                continue
            trade_cost = market.calculate_trade_cost(x_star)  # LMSR price and inventory update
            agent.update_portfolio(x_star, trade_cost)
            round_volume += abs(x_star)

        trade_volume.append(round_volume)

    mean_initial_belief = float(np.mean(beliefs))
    final_price = float(price_series[-1]) if price_series else float(market.get_price())
    # Compute time series and final error between market price and ground truth
    error_series = [abs(q - ground_truth) for q in price_series]
    final_error = float(error_series[-1]) if error_series else abs(final_price - ground_truth)

    # Gather agent outcomes for export and analysis
    final_positions = [agent.shares for agent in agents]
    final_cash = [agent.cash for agent in agents]
    final_rhos = [agent.rho for agent in agents]

    # Aggregate agent positions and cash by rho value for summary statistics
    rho_groups = {}
    for rho, shares, cash in zip(final_rhos, final_positions, final_cash):
        if rho not in rho_groups:
            rho_groups[rho] = {"shares": [], "cash": []}
        rho_groups[rho]["shares"].append(shares)
        rho_groups[rho]["cash"].append(cash)

    rho_summary = {}
    for rho, data in rho_groups.items():
        rho_summary[rho] = {
            "avg_shares": float(np.mean(data["shares"])) if data["shares"] else 0.0,
            "avg_cash": float(np.mean(data["cash"])) if data["cash"] else 0.0,
        }

    # Return structure for downstream export and reporting
    return {
        "seed": seed,
        "ground_truth": ground_truth,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "b": b,
        "mean_initial_belief": mean_initial_belief,
        "final_price": final_price,
        "final_error": final_error,
        "price_series": price_series,
        "error_series": error_series,
        "trade_volume": trade_volume,
        "final_positions": final_positions,
        "final_cash": final_cash,
        "final_rhos": final_rhos,
        "rho_summary": rho_summary,
    }
