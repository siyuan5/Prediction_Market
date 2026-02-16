import numpy as np


class CRRAAgent:
    def __init__(self, agent_id, initial_cash, belief_p, rho):
        self.id = agent_id
        self.cash = initial_cash  # y in the paper
        self.shares = 0          # z in the paper
        self.belief = belief_p   # p in the paper
        self.rho = rho           # Risk aversion parameter

    def get_optimal_trade(self, market_price):
        """
        Calculates optimal trade size x* based on Eq 8 in Sethi et al. (2024).

        Returns:
            x_star (float): Number of shares to buy (+) or sell (-).
        """
        q = market_price
        p = self.belief
        y = self.cash
        z = self.shares
        rho = self.rho

        # Avoid division by zero or log errors
        if q <= 0.01 or q >= 0.99:
            return 0.0

        # Case 1: Agent agrees with market (No trade)
        if abs(p - q) < 1e-6:
            # The paper notes that when p=q, the agent liquidates (x = -z).
            # However, for stability, we can simply hold (return 0).
            return 0.0

        # Calculate k (The risk-weighted edge)
        # k = ((p * (1 - q)) / (q * (1 - p))) ^ (1 / rho)
        numerator = p * (1 - q)
        denominator = q * (1 - p)
        k = (numerator / denominator) ** (1 / rho)

        # Calculate x* (Optimal Trade)
        # x* = ((k - 1) * y - z) / (1 + q * (k - 1))
        x_star = ((k - 1) * y - z) / (1 + q * (k - 1))

        # --- SAFETY CHECKS (Bankruptcy Constraints) ---
        # Ensure the trade doesn ’t result in negative wealth in either outcome.

        # Max buy (limited by cash)
        max_buy = y / q if q > 0 else 0

        # Max sell (limited by "margin" - simplified)
        if x_star > 0:
            x_star = min(x_star, max_buy)
        else:
            max_sell = y / (1 - q) if (1 - q) > 0 else 0
            x_star = max(x_star, -max_sell)

        return x_star

    def update_portfolio(self, trade_shares, trade_cost):
        """
        Updates cash and shares after a trade is executed.

        Args:
            trade_shares (float): Number of shares bought (+) or sold (-).
            trade_cost (float): Total cost of the trade from the market maker.
        """
        self.cash -= trade_cost
        self.shares += trade_shares

    def update_belief(self, signal_s, *, method="beta", w=0.10, prior_strength=20.0, obs_strength=5.0):
        """
        Phase 2: update internal belief using public noisy signal S_t.

        method:
          - "weighted": p <- (1-w)p + w*S_t
          - "beta": Beta pseudo-count update (stable + interpretable)
        """
        from phase2_utils import update_belief_weighted, update_belief_beta

        if method == "weighted":
            self.belief = update_belief_weighted(self.belief, float(signal_s), float(w))
        elif method == "beta":
            self.belief = update_belief_beta(
                self.belief, float(signal_s),
                prior_strength=float(prior_strength),
                obs_strength=float(obs_strength),
            )
        else:
            raise ValueError(f"Unknown method={method!r}")
