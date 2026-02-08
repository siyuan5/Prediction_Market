import numpy as np

from team_b_crra_agent import TeamBCRRAAgent
from team_b_market_logic import ContinuousDoubleAuction


def clipped_gaussian(mean, sigma, size, low=0.01, high=0.99, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.normal(loc=mean, scale=sigma, size=size)
    return np.clip(samples, low, high)


def _apply_trades(trades, agents_by_id):
    volume = 0.0
    for trade in trades:
        quantity = trade.quantity
        notional = trade.price * quantity

        # Exchange outputs matched counterparties; portfolios are updated symmetrically.
        buyer = agents_by_id[trade.buyer_id]
        seller = agents_by_id[trade.seller_id]

        buyer.update_portfolio(trade_shares=quantity, trade_cost=notional)
        seller.update_portfolio(trade_shares=-quantity, trade_cost=-notional)
        volume += quantity
    return volume


def run_team_b_phase1(
    *,
    seed=42,
    ground_truth=0.70,
    n_agents=50,
    n_rounds=500,
    initial_cash=100.0,
    sigma=0.10,
    rho_values=None,
    fixed_rho=None,
    convergence_tol=1e-2,
    stable_rounds=5,
    min_trade_size=1e-6,
    max_idle_rounds=25,
    shuffle_agents=True,
    initial_price=0.5,
    tick_size=1e-4,
    order_policy="hybrid",
    limit_offset=0.01,
    market_order_edge=0.08,
):
    rng = np.random.default_rng(seed)

    if rho_values is None:
        rho_values = [0.5, 1.0, 2.0]

    beliefs = clipped_gaussian(ground_truth, sigma, n_agents, rng=rng)
    if fixed_rho is None:
        rhos = rng.choice(rho_values, size=n_agents, replace=True)
    else:
        rhos = np.full(n_agents, float(fixed_rho))

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

    exchange = ContinuousDoubleAuction(
        tick_size=tick_size,
        initial_reference_price=initial_price,
    )
    mean_initial_belief = float(np.mean(beliefs))

    price_series = [exchange.reference_price()]
    trade_volume = []
    trade_count = []
    stable_count = 0
    idle_rounds = 0
    rounds_run = 0

    for _ in range(n_rounds):
        rounds_run += 1
        round_volume = 0.0
        round_trade_count = 0

        if shuffle_agents:
            order = rng.permutation(len(agents))
        else:
            order = range(len(agents))

        for idx in order:
            agent = agents[idx]
            # Keep at most one active quote per agent each step.
            exchange.cancel_agent_orders(agent.id)

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
                continue

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

            trades = result["trades"]
            round_trade_count += len(trades)
            round_volume += _apply_trades(trades, agents_by_id)

        trade_volume.append(round_volume)
        trade_count.append(round_trade_count)

        final_price = exchange.reference_price()
        price_series.append(final_price)

        if abs(final_price - mean_initial_belief) <= convergence_tol:
            stable_count += 1
        else:
            stable_count = 0

        # Stop either on sustained convergence or when trading dries up.
        if round_volume <= min_trade_size:
            idle_rounds += 1
        else:
            idle_rounds = 0

        if stable_count >= stable_rounds:
            break
        if idle_rounds >= max_idle_rounds:
            break

    final_positions = [agent.shares for agent in agents]
    final_cash = [agent.cash for agent in agents]
    final_rhos = [agent.rho for agent in agents]
    final_price = price_series[-1]
    converged = abs(final_price - mean_initial_belief) <= convergence_tol

    return {
        "seed": seed,
        "ground_truth": ground_truth,
        "n_agents": n_agents,
        "n_rounds_requested": n_rounds,
        "rounds_run": rounds_run,
        "mean_initial_belief": mean_initial_belief,
        "final_price": final_price,
        "converged": converged,
        "convergence_tol": convergence_tol,
        "price_series": price_series,
        "trade_volume": trade_volume,
        "trade_count": trade_count,
        "final_positions": final_positions,
        "final_cash": final_cash,
        "final_rhos": final_rhos,
        "order_policy": order_policy,
        "limit_offset": limit_offset,
        "market_order_edge": market_order_edge,
    }


def analyze_team_b_rho_effect(
    *,
    rho_values,
    n_seeds=20,
    **phase1_kwargs,
):
    analysis = []

    for rho in rho_values:
        abs_positions = []
        final_price_gaps = []
        convergence_hits = 0
        rounds = []
        total_trade_volume = []

        for seed in range(n_seeds):
            result = run_team_b_phase1(
                seed=seed,
                fixed_rho=float(rho),
                rho_values=[float(rho)],
                **phase1_kwargs,
            )
            positions = np.asarray(result["final_positions"], dtype=float)
            abs_positions.append(np.mean(np.abs(positions)))
            final_price_gaps.append(
                abs(result["final_price"] - result["mean_initial_belief"])
            )
            rounds.append(result["rounds_run"])
            total_trade_volume.append(float(np.sum(result["trade_volume"])))
            if result["converged"]:
                convergence_hits += 1

        analysis.append(
            {
                "rho": float(rho),
                "mean_abs_position": float(np.mean(abs_positions)),
                "std_abs_position": float(np.std(abs_positions)),
                "mean_final_price_gap": float(np.mean(final_price_gaps)),
                "convergence_rate": convergence_hits / n_seeds,
                "mean_rounds": float(np.mean(rounds)),
                "mean_total_volume": float(np.mean(total_trade_volume)),
            }
        )

    return analysis
