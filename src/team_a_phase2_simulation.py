"""
Standalone Phase 2 LMSR simulation (legacy script path).

Prefer `SimulationEngine(mechanism="lmsr", phase=2, ...)` for new code; this module
keeps the original `run_phase2` API used by `run_team_a_phase2.py` and tests.
"""

import numpy as np

from crra_agent import CRRAAgent
from phase2_utils import SignalSpec, generate_signal
from team_a_market_logic import LMSRMarketMaker

def clipped_gaussian(mean, sigma, size, low=0.01, high=0.99, rng=None):
    """Draw normal samples near mean, clip to (low, high).
    Used for initializing agent beliefs away from edge values."""
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(loc=mean, scale=sigma, size=size)
    return np.clip(samples, low, high)

def run_phase2(
    *,
    seed=42,
    ground_truth=0.70,
    n_agents=50,
    n_rounds=100,
    initial_cash=100.0,
    sigma=0.10,
    rho_values=None,
    b=100.0,
    signal_spec=None,
    belief_update_method="beta",
    belief_weight=0.10,
    prior_strength=20.0,
    obs_strength=10.0,
    min_trade_size=1e-9,
    shuffle_agents=False,
    trade_fraction=0.20,
):
    # Set fixed seed for reproducibility
    rng = np.random.default_rng(seed)

    # Default risk aversion values if not provided
    if rho_values is None:
        rho_values = [0.5, 1.0, 2.0]

    # Default signal spec: binomial draws, 25 trials per round
    if signal_spec is None:
        signal_spec = SignalSpec(mode="binomial", n=25)

    # Initialize agent beliefs and risk aversion
    beliefs = clipped_gaussian(ground_truth, sigma, n_agents, rng=rng)
    rhos = rng.choice(rho_values, size=n_agents, replace=True)

    # Instantiate all CRRA agents with initial values
    agents = [
        CRRAAgent(
            agent_id=i,
            initial_cash=initial_cash,
            belief_p=float(beliefs[i]),
            rho=float(rhos[i]),
        )
        for i in range(n_agents)
    ]

    # Set up LMSR market maker (liquidity parameter b)
    market = LMSRMarketMaker(b=b)
    mean_initial_belief = float(np.mean(beliefs))

    # Initialize series for tracking results during simulation
    price_series = []
    signal_series = []
    mean_belief_series = []
    error_series = []
    trade_volume = []
    inventory_series = []

    for _ in range(n_rounds):
        # Draw a public signal (same for all agents this round)
        signal_t = generate_signal(ground_truth, rng, signal_spec)
        signal_series.append(float(signal_t))

        # Each agent updates belief based on signal (using chosen update method)
        for agent in agents:
            agent.update_belief(
                signal_t,
                method=belief_update_method,
                w=belief_weight,
                prior_strength=prior_strength,
                obs_strength=obs_strength,
            )

        round_volume = 0.0

        # Shuffle agent order if enabled (simulate asynchronous trade order)
        if shuffle_agents:
            order = rng.permutation(len(agents))
        else:
            order = range(len(agents))

        for idx in order:
            agent = agents[idx]
            # Agent sees current price (after any prior trades this round)
            q_t = market.get_price()
            # Compute optimal trade (crra_agent handles core math)
            x_star = agent.get_optimal_trade(q_t) * trade_fraction  # trade_fraction dampens aggressiveness
            if abs(x_star) < min_trade_size:
                continue  # skip negligible trades
            trade_cost = market.calculate_trade_cost(x_star)  # market applies trade & updates inventory
            agent.update_portfolio(x_star, trade_cost)
            round_volume += abs(x_star)

        # Track post-round statistics
        q_next = float(market.get_price())
        mean_belief = float(np.mean([agent.belief for agent in agents]))

        price_series.append(q_next)
        mean_belief_series.append(mean_belief)
        error_series.append(abs(q_next - ground_truth))
        trade_volume.append(round_volume)
        inventory_series.append(list(market.inventory.copy()))

    # Use last price, or fallback to current price if simulation ran zero rounds
    final_price = float(price_series[-1]) if price_series else float(market.get_price())
    final_positions = [agent.shares for agent in agents]
    final_cash = [agent.cash for agent in agents]
    final_rhos = [agent.rho for agent in agents]
    final_beliefs = [agent.belief for agent in agents]

    # Return time series and summary stats for downstream analysis
    return {
        "seed": seed,
        "ground_truth": ground_truth,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "b": b,
        "mean_initial_belief": mean_initial_belief,
        "final_price": final_price,
        "final_error": abs(final_price - ground_truth),
        "price_series": price_series,
        "signal_series": signal_series,
        "mean_belief_series": mean_belief_series,
        "error_series": error_series,
        "trade_volume": trade_volume,
        "inventory_series": inventory_series,
        "signal_mode": signal_spec.mode,
        "belief_update_method": belief_update_method,
        "final_positions": final_positions,
        "final_cash": final_cash,
        "final_rhos": final_rhos,
        "final_beliefs": final_beliefs,
    }
