import numpy as np

from crra_agent import CRRAAgent
from market_logic import LMSRMarketMaker


def clipped_gaussian(mean, sigma, size, low=0.01, high=0.99, rng=None):
    # sample beliefs around ground truth, clipped to avoid 0/1
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
    # fixed seed for repeatable runs
    rng = np.random.default_rng(seed)

    if rho_values is None:
        # simple mix of risk levels
        rho_values = [0.5, 1.0, 2.0]

    # static beliefs for phase 1
    beliefs = clipped_gaussian(ground_truth, sigma, n_agents, rng=rng)
    rhos = rng.choice(rho_values, size=n_agents, replace=True)

    agents = [
        CRRAAgent(
            agent_id=i,
            initial_cash=initial_cash,
            belief_p=float(beliefs[i]),
            rho=float(rhos[i]),
        )
        for i in range(n_agents)
    ]

    # lmsr market maker
    market = LMSRMarketMaker(b=b)

    price_series = []
    trade_volume = []

    for _ in range(n_rounds):
        # current market price before trades
        q_t = market.get_price()
        price_series.append(q_t)

        round_volume = 0.0
        for agent in agents:
            x_star = agent.get_optimal_trade(q_t)
            if abs(x_star) < 1e-9:
                continue
            trade_cost = market.calculate_trade_cost(x_star)
            agent.update_portfolio(x_star, trade_cost)
            round_volume += abs(x_star)

        trade_volume.append(round_volume)

    mean_initial_belief = float(np.mean(beliefs))

    final_positions = [agent.shares for agent in agents]
    final_cash = [agent.cash for agent in agents]
    final_rhos = [agent.rho for agent in agents]

    return {
        "seed": seed,
        "ground_truth": ground_truth,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "b": b,
        "mean_initial_belief": mean_initial_belief,
        "price_series": price_series,
        "trade_volume": trade_volume,
        "final_positions": final_positions,
        "final_cash": final_cash,
        "final_rhos": final_rhos,
    }
