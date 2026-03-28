# stateful sim engine for both lmsr and cda
# lets you run in chunks, shift beliefs mid-run, and poll state
#
# how to use:
#   engine = SimulationEngine(mechanism="lmsr", phase=2, ground_truth=0.70)
#   engine.run(30)                          # run 30 rounds
#   engine.shift_beliefs(new_belief=0.9)    # shock all agents
#   engine.run(20)                          # keep going
#   engine.get_state()                      # current snapshot
#   engine.get_agents()                     # per-agent data
#   engine.get_metrics()                    # full history

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from .crra_agent import CRRAAgent
    from .team_b_crra_agent import TeamBCRRAAgent
    from .team_a_market_logic import LMSRMarketMaker
    from .team_b_market_logic import ContinuousDoubleAuction
    from .phase2_utils import SignalSpec, generate_signal
    from .belief_init import BeliefSpec, sample_beliefs
except ImportError:
    from crra_agent import CRRAAgent
    from team_b_crra_agent import TeamBCRRAAgent
    from team_a_market_logic import LMSRMarketMaker
    from team_b_market_logic import ContinuousDoubleAuction
    from phase2_utils import SignalSpec, generate_signal
    from belief_init import BeliefSpec, sample_beliefs


class SimulationEngine:
    # mechanism: "lmsr" or "cda"
    # phase 1 = static beliefs, phase 2 = agents get signals each round
    # trade_fraction defaults to 1.0 for phase 1, 0.20 for phase 2

    def __init__(
        self,
        *,
        mechanism: str = "lmsr",
        phase: int = 1,
        seed: int = 42,
        ground_truth: float = 0.70,
        n_agents: int = 50,
        initial_cash: float = 100.0,
        rho_values: Optional[List[float]] = None,
        belief_spec: Optional[BeliefSpec] = None,
        # lmsr
        b: float = 100.0,
        # cda
        initial_price: float = 0.5,
        tick_size: float = 1e-4,
        order_policy: str = "hybrid",
        limit_offset: float = 0.01,
        market_order_edge: float = 0.08,
        # phase 2 signals
        signal_spec: Optional[SignalSpec] = None,
        belief_update_method: str = "beta",
        belief_weight: float = 0.10,
        prior_strength: float = 20.0,
        obs_strength: float = 10.0,
        # execution
        trade_fraction: Optional[float] = None,
        min_trade_size: float = 1e-9,
        shuffle_agents: bool = True,
    ):
        if mechanism not in ("lmsr", "cda"):
            raise ValueError(f"mechanism must be 'lmsr' or 'cda', got {mechanism!r}")
        if phase not in (1, 2):
            raise ValueError(f"phase must be 1 or 2, got {phase!r}")

        self.mechanism = mechanism
        self.phase = phase
        self.ground_truth = ground_truth
        self.n_agents = n_agents
        self.initial_cash = initial_cash
        self.belief_update_method = belief_update_method
        self.belief_weight = belief_weight
        self.prior_strength = prior_strength
        self.obs_strength = obs_strength
        self.min_trade_size = min_trade_size
        self.shuffle_agents = shuffle_agents
        self.order_policy = order_policy
        self.limit_offset = limit_offset
        self.market_order_edge = market_order_edge

        # full trade in phase 1, fractional in phase 2 to avoid oscillation
        if trade_fraction is None:
            self.trade_fraction = 1.0 if phase == 1 else 0.20
        else:
            self.trade_fraction = trade_fraction

        self.rng = np.random.default_rng(seed)

        if rho_values is None:
            rho_values = [0.5, 1.0, 2.0]
        if belief_spec is None:
            belief_spec = BeliefSpec()
        if signal_spec is None:
            signal_spec = SignalSpec(mode="binomial", n=25)

        self.signal_spec = signal_spec

        # build agents
        beliefs = sample_beliefs(ground_truth, n_agents, belief_spec, self.rng)
        rhos = self.rng.choice(rho_values, size=n_agents, replace=True)

        if mechanism == "lmsr":
            self.agents = [
                CRRAAgent(
                    agent_id=i,
                    initial_cash=initial_cash,
                    belief_p=float(beliefs[i]),
                    rho=float(rhos[i]),
                )
                for i in range(n_agents)
            ]
        else:
            self.agents = [
                TeamBCRRAAgent(
                    agent_id=i,
                    initial_cash=initial_cash,
                    belief_p=float(beliefs[i]),
                    rho=float(rhos[i]),
                )
                for i in range(n_agents)
            ]

        self.agents_by_id: Dict[int, Any] = {a.id: a for a in self.agents}
        self.initial_beliefs: List[float] = beliefs.tolist()
        self.mean_initial_belief: float = float(np.mean(beliefs))

        # spin up market
        if mechanism == "lmsr":
            self.market: Any = LMSRMarketMaker(b=b)
        else:
            self.market = ContinuousDoubleAuction(
                tick_size=tick_size,
                initial_reference_price=initial_price,
            )

        # full history across all segments
        self.round: int = 0
        self.price_series: List[float] = []
        self.error_series: List[float] = []
        self.trade_volume: List[float] = []
        self.mean_belief_series: List[float] = []
        self.signal_series: List[float] = []
        self.belief_shift_events: List[Dict[str, Any]] = []
        # cda only
        self.best_bid_series: List[Optional[float]] = []
        self.best_ask_series: List[Optional[float]] = []

    def run(self, n_rounds: int) -> Dict[str, Any]:
        # run n more rounds, state carries over between calls
        # returns metrics for this segment only
        seg_prices: List[float] = []
        seg_errors: List[float] = []
        seg_volume: List[float] = []
        seg_mean_beliefs: List[float] = []
        seg_signals: List[float] = []

        for _ in range(n_rounds):
            self.round += 1

            # phase 2: emit signal, all agents update before trading
            if self.phase == 2:
                signal_t = float(generate_signal(self.ground_truth, self.rng, self.signal_spec))
                self.signal_series.append(signal_t)
                seg_signals.append(signal_t)
                for agent in self.agents:
                    agent.update_belief(
                        signal_t,
                        method=self.belief_update_method,
                        w=self.belief_weight,
                        prior_strength=self.prior_strength,
                        obs_strength=self.obs_strength,
                    )

            round_volume = self._run_round()
            price_t = self._current_price()
            mean_belief_t = float(np.mean([a.belief for a in self.agents]))
            error_t = abs(price_t - self.ground_truth)

            self.price_series.append(price_t)
            self.error_series.append(error_t)
            self.trade_volume.append(round_volume)
            self.mean_belief_series.append(mean_belief_t)
            seg_prices.append(price_t)
            seg_errors.append(error_t)
            seg_volume.append(round_volume)
            seg_mean_beliefs.append(mean_belief_t)

            if self.mechanism == "cda":
                self.best_bid_series.append(self.market.best_bid())
                self.best_ask_series.append(self.market.best_ask())

        return {
            "rounds_run": n_rounds,
            "price_series": seg_prices,
            "error_series": seg_errors,
            "trade_volume": seg_volume,
            "mean_belief_series": seg_mean_beliefs,
            "signal_series": seg_signals,
            "final_price": seg_prices[-1] if seg_prices else None,
        }

    def shift_beliefs(
        self,
        *,
        new_belief: Optional[float] = None,
        delta: Optional[float] = None,
        agent_ids: Optional[List[int]] = None,
        rho_filter: Optional[float] = None,
    ) -> Dict[str, Any]:
        # mid-run belief injection
        # pass new_belief to set absolute value, or delta to nudge relative
        # filter by agent_ids or rho_filter to target a subset
        if (new_belief is None) == (delta is None):
            raise ValueError("pass exactly one of new_belief or delta")

        targets = list(self.agents)

        if agent_ids is not None:
            id_set = set(agent_ids)
            targets = [a for a in targets if a.id in id_set]

        if rho_filter is not None:
            targets = [a for a in targets if abs(a.rho - rho_filter) < 1e-9]

        before_mean = float(np.mean([a.belief for a in targets])) if targets else 0.0

        for agent in targets:
            if new_belief is not None:
                agent.belief = float(np.clip(new_belief, 0.01, 0.99))
            else:
                assert delta is not None
                agent.belief = float(np.clip(agent.belief + delta, 0.01, 0.99))

        after_mean = float(np.mean([a.belief for a in targets])) if targets else 0.0

        # log the event so the gui can mark it on the chart
        event: Dict[str, Any] = {
            "round": self.round,
            "n_agents_shifted": len(targets),
            "agent_ids": [a.id for a in targets],
            "new_belief": new_belief,
            "delta": delta,
            "before_mean": before_mean,
            "after_mean": after_mean,
        }
        self.belief_shift_events.append(event)
        return event

    def get_state(self) -> Dict[str, Any]:
        # current round snapshot, used by api /state
        price = self._current_price()
        return {
            "round": self.round,
            "price": price,
            "error": abs(price - self.ground_truth),
            "mean_belief": float(np.mean([a.belief for a in self.agents])),
            "ground_truth": self.ground_truth,
            "mechanism": self.mechanism,
            "phase": self.phase,
            "belief_shift_events": self.belief_shift_events,
        }

    def get_agents(self) -> List[Dict[str, Any]]:
        # per-agent snapshot, used by api /agents
        price = self._current_price()
        return [
            {
                "agent_id": a.id,
                "belief": a.belief,
                "rho": a.rho,
                "cash": a.cash,
                "shares": a.shares,
                "pnl": a.cash + a.shares * price - self.initial_cash,
            }
            for a in self.agents
        ]

    def get_metrics(self) -> Dict[str, Any]:
        # full history across all segments, used by api /metrics
        result: Dict[str, Any] = {
            "total_rounds": self.round,
            "price_series": self.price_series,
            "error_series": self.error_series,
            "trade_volume": self.trade_volume,
            "mean_belief_series": self.mean_belief_series,
            "signal_series": self.signal_series,
            "belief_shift_events": self.belief_shift_events,
            "mean_initial_belief": self.mean_initial_belief,
            "final_price": self.price_series[-1] if self.price_series else None,
            "final_error": self.error_series[-1] if self.error_series else None,
        }
        if self.mechanism == "cda":
            result["best_bid_series"] = self.best_bid_series
            result["best_ask_series"] = self.best_ask_series
        return result

    # internal helpers

    def _current_price(self) -> float:
        if self.mechanism == "lmsr":
            return float(self.market.get_price())
        return float(self.market.reference_price())

    def _run_round(self) -> float:
        if self.mechanism == "lmsr":
            return self._run_lmsr_round()
        return self._run_cda_round()

    def _run_lmsr_round(self) -> float:
        # refresh price inside loop so each agent trades at post-trade price
        order = (
            self.rng.permutation(len(self.agents))
            if self.shuffle_agents
            else range(len(self.agents))
        )
        volume = 0.0
        for idx in order:
            agent = self.agents[idx]
            q_t = self.market.get_price()
            x_star = agent.get_optimal_trade(q_t) * self.trade_fraction
            if abs(x_star) < self.min_trade_size:
                continue
            trade_cost = self.market.calculate_trade_cost(x_star)
            agent.update_portfolio(x_star, trade_cost)
            volume += abs(x_star)
        return volume

    def _run_cda_round(self) -> float:
        # cancel stale quotes first, then resubmit based on updated belief
        order = (
            self.rng.permutation(len(self.agents))
            if self.shuffle_agents
            else range(len(self.agents))
        )
        volume = 0.0
        for idx in order:
            agent = self.agents[idx]
            self.market.cancel_agent_orders(agent.id)
            ref = self.market.reference_price()
            order_spec = agent.build_order(
                reference_price=ref,
                best_bid=self.market.best_bid(),
                best_ask=self.market.best_ask(),
                order_policy=self.order_policy,
                limit_offset=self.limit_offset,
                market_order_edge=self.market_order_edge,
                min_trade_size=self.min_trade_size,
            )
            if order_spec is None:
                continue

            if order_spec["type"] == "market":
                result = self.market.submit_market_order(
                    agent_id=agent.id,
                    side=order_spec["side"],
                    quantity=order_spec["quantity"],
                )
            else:
                result = self.market.submit_limit_order(
                    agent_id=agent.id,
                    side=order_spec["side"],
                    quantity=order_spec["quantity"],
                    limit_price=order_spec["limit_price"],
                )

            for trade in result["trades"]:
                qty = trade.quantity
                notional = trade.price * qty
                buyer = self.agents_by_id[trade.buyer_id]
                seller = self.agents_by_id[trade.seller_id]
                buyer.update_portfolio(trade_shares=qty, trade_cost=notional)
                seller.update_portfolio(trade_shares=-qty, trade_cost=-notional)
                volume += qty
        return volume
