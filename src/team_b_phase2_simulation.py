"""
Phase 2 CDA simulation: signals, belief updates, then order-book trading each round.

`SimulationEngine` wraps the same ideas; this file is the older explicit loop for Team B.
"""

import numpy as np

# Support both relative and direct imports so module runs both as script and package
try:
    from .phase2_utils import SignalSpec, generate_signal
    from .team_b_crra_agent import TeamBCRRAAgent
    from .team_b_market_logic import ContinuousDoubleAuction
except ImportError:
    from phase2_utils import SignalSpec, generate_signal
    from team_b_crra_agent import TeamBCRRAAgent
    from team_b_market_logic import ContinuousDoubleAuction


def clipped_gaussian(mean, sigma, size, low=0.01, high=0.99, rng=None):
    """
    Generate samples from a Gaussian clipped to [low, high].
    Defaults to using a new random generator if none supplied.
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(loc=mean, scale=sigma, size=size)
    return np.clip(samples, low, high)


def _apply_trades(trades, agents_by_id):
    """
    Update agent portfolios according to executed trades and return total round volume.
    """
    volume = 0.0
    for trade in trades:
        quantity = trade.quantity
        notional = trade.price * quantity

        buyer = agents_by_id[trade.buyer_id]
        seller = agents_by_id[trade.seller_id]

        # Update buyer and seller portfolios based on trade direction
        buyer.update_portfolio(trade_shares=quantity, trade_cost=notional)
        seller.update_portfolio(trade_shares=-quantity, trade_cost=-notional)
        volume += quantity
    return volume


def run_team_b_phase2(
    *,
    seed=42,
    ground_truth=0.70,
    n_agents=50,
    n_rounds=100,
    initial_cash=100.0,
    sigma=0.10,
    rho_values=None,
    fixed_rho=None,
    signal_spec=None,
    belief_update_method="beta",
    belief_weight=0.10,
    prior_strength=20.0,
    obs_strength=5.0,
    min_trade_size=1e-6,
    max_idle_rounds=None,
    shuffle_agents=True,
    initial_price=0.5,
    tick_size=1e-4,
    order_policy="hybrid",
    limit_offset=0.01,
    market_order_edge=0.08,
):
    """
    Simulates Phase 2: dynamic beliefs with signals and belief-updates.
    Runs a multi-agent CDA with CRRA agents updating beliefs in response to noisy signals.
    """
    rng = np.random.default_rng(seed)

    # Set up heterogeneity in CRRA parameter ("risk aversion"), if not fixed
    if rho_values is None:
        rho_values = [0.5, 1.0, 2.0]
    # Default: binomial signal of size 25 (user can override via signal_spec)
    if signal_spec is None:
        signal_spec = SignalSpec(mode="binomial", n=25)

    # Each agent starts with a different (clipped normal) initial belief centered at ground truth
    beliefs = clipped_gaussian(ground_truth, sigma, n_agents, rng=rng)
    if fixed_rho is None:
        rhos = rng.choice(rho_values, size=n_agents, replace=True)
    else:
        rhos = np.full(n_agents, float(fixed_rho))

    # Instantiate CRRA agents for each participant
    agents = [
        TeamBCRRAAgent(
            agent_id=i,
            initial_cash=initial_cash,
            belief_p=float(beliefs[i]),
            rho=float(rhos[i]),
        )
        for i in range(n_agents)
    ]
    agents_by_id = {agent.id: agent for agent in agents}

    # Initialize CDA exchange with given tick size/initial price
    exchange = ContinuousDoubleAuction(
        tick_size=tick_size,
        initial_reference_price=initial_price,
    )
    mean_initial_belief = float(np.mean(beliefs))

    # Track time series and outcomes for diagnostics/metrics
    price_series = []
    signal_series = []
    mean_belief_series = []
    error_series = []
    trade_volume = []
    trade_count = []
    best_bid_series = []
    best_ask_series = []
    mid_price_series = []

    idle_rounds = 0     # Counts consecutive rounds with < min_trade_size volume
    rounds_run = 0

    for _ in range(n_rounds):
        rounds_run += 1

        # Sample and store the round's noisy signal
        signal_t = float(generate_signal(ground_truth, rng, signal_spec))
        signal_series.append(signal_t)

        # All agents update their beliefs in response to this signal
        for agent in agents:
            agent.update_belief(
                signal_t,
                method=belief_update_method,
                w=belief_weight,
                prior_strength=prior_strength,
                obs_strength=obs_strength,
            )

        round_volume = 0.0
        round_trade_count = 0

        # Trading occurs in random (or fixed) sequence to reduce order effects
        if shuffle_agents:
            order = rng.permutation(len(agents))
        else:
            order = range(len(agents))

        for idx in order:
            agent = agents[idx]
            # Cancel the agent's existing resting quote, so only one open order at a time
            exchange.cancel_agent_orders(agent.id)

            # Compute reference price and agent's optimal order for this round
            reference_price = exchange.reference_price()
            order_spec = agent.build_order(
                reference_price=reference_price,
                best_bid=exchange.best_bid(),
                best_ask=exchange.best_ask(),
                order_policy=order_policy,
                limit_offset=limit_offset,
                market_order_edge=market_order_edge,
                min_trade_size=min_trade_size,
            )
            if order_spec is None:
                continue  # Agent chooses not to submit any order

            # Submit market or limit order as specified; collect execution result
            if order_spec["type"] == "market":
                result = exchange.submit_market_order(
                    agent_id=agent.id,
                    side=order_spec["side"],
                    quantity=order_spec["quantity"],
                )
            else:
                result = exchange.submit_limit_order(
                    agent_id=agent.id,
                    side=order_spec["side"],
                    quantity=order_spec["quantity"],
                    limit_price=order_spec["limit_price"],
                )

            # Apply trades to agent portfolios, accumulate stats
            trades = result["trades"]
            round_trade_count += len(trades)
            round_volume += _apply_trades(trades, agents_by_id)

        # Track price/belief/volume and spread series for analysis/plotting
        price_t = float(exchange.reference_price())
        mean_belief_t = float(np.mean([agent.belief for agent in agents]))

        price_series.append(price_t)
        mean_belief_series.append(mean_belief_t)
        error_series.append(abs(price_t - ground_truth))
        trade_volume.append(round_volume)
        trade_count.append(round_trade_count)
        best_bid_series.append(exchange.best_bid())
        best_ask_series.append(exchange.best_ask())
        mid_price_series.append(exchange.mid_price())

        # Idle round if trading dried up to below min_trade_size
        if round_volume <= min_trade_size:
            idle_rounds += 1
        else:
            idle_rounds = 0

        # Early stop if there are too many consecutive idle (illiquid) rounds
        if max_idle_rounds is not None and idle_rounds >= int(max_idle_rounds):
            break

    # Record end-of-simulation stats for all agents and output series
    final_price = float(price_series[-1]) if price_series else float(exchange.reference_price())
    final_positions = [agent.shares for agent in agents]
    final_cash = [agent.cash for agent in agents]
    final_rhos = [agent.rho for agent in agents]
    final_beliefs = [agent.belief for agent in agents]

    return {
        "seed": seed,
        "ground_truth": ground_truth,
        "n_agents": n_agents,
        "n_rounds_requested": n_rounds,
        "rounds_run": rounds_run,
        "mean_initial_belief": mean_initial_belief,
        "initial_beliefs": [float(x) for x in beliefs.tolist()],
        "final_price": final_price,
        "final_error": abs(final_price - ground_truth),
        "price_series": price_series,
        "signal_series": signal_series,
        "mean_belief_series": mean_belief_series,
        "error_series": error_series,
        "trade_volume": trade_volume,
        "trade_count": trade_count,
        "best_bid_series": best_bid_series,
        "best_ask_series": best_ask_series,
        "mid_price_series": mid_price_series,
        "signal_mode": getattr(signal_spec, "mode", None),
        "belief_update_method": belief_update_method,
        "order_policy": order_policy,
        "limit_offset": limit_offset,
        "market_order_edge": market_order_edge,
        "final_positions": final_positions,
        "final_cash": final_cash,
        "final_rhos": final_rhos,
        "final_beliefs": final_beliefs,
    }


def estimate_required_rounds_phase2(
    *,
    n_seeds=20,
    n_rounds_cap=200,
    error_threshold=0.05,
    percentile=95,
    **phase2_kwargs,
):
    """
    Runs Phase 2 simulation repeatedly to estimate rounds needed for price to reach
    within error_threshold of ground truth (at least once), e.g., convergence speed.
    Returns stats and a recommended round count for experiment design.
    """
    rounds_to_threshold = []
    final_errors = []
    for seed in range(n_seeds):
        result = run_team_b_phase2(
            seed=seed,
            n_rounds=n_rounds_cap,
            **phase2_kwargs,
        )
        err_series = result["error_series"]
        final_errors.append(result["final_error"])
        # Find first round (1-based) when price error becomes acceptable
        hit = None
        for t, e in enumerate(err_series):
            if e <= error_threshold:
                hit = t + 1
                break
        if hit is not None:
            rounds_to_threshold.append(hit)

    # If price never hit the threshold in any run, fill with cap for safety
    rounds_arr = np.asarray(rounds_to_threshold) if rounds_to_threshold else np.array([n_rounds_cap])
    recommended = int(np.percentile(rounds_arr, percentile))
    return {
        "mean_rounds_to_threshold": float(np.mean(rounds_arr)),
        "std_rounds_to_threshold": float(np.std(rounds_arr)) if len(rounds_arr) > 1 else 0.0,
        "percentile_rounds": recommended,
        "percentile_used": percentile,
        "error_threshold": error_threshold,
        "runs_that_hit_threshold": len(rounds_to_threshold),
        "n_seeds": n_seeds,
        "mean_final_error": float(np.mean(final_errors)),
        "recommended_n_rounds": min(recommended, n_rounds_cap),
    }

