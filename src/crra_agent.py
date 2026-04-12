"""
CRRA (constant relative risk aversion) agent trading against the LMSR in Team A runs.

Optimal trade size follows Sethi et al. (2024); `rho` controls risk aversion.
Used with `LMSRMarketMaker` inside `SimulationEngine` (mechanism=lmsr).
"""

from crra_math import compute_optimal_trade


class CRRAAgent:
    """Single agent: cash, YES shares, belief p ∈ (0,1), and risk aversion rho > 0."""

    def __init__(self, agent_id, initial_cash, belief_p, rho, *,
                 prior_strength=20.0, obs_strength=10.0, participation_rate=1.0):
        self.id = agent_id
        self.cash = initial_cash  # y in the paper
        self.shares = 0          # z in the paper
        self.belief = belief_p   # p in the paper
        self.rho = rho           # Risk aversion parameter
        # Per-agent personality: stubbornness and signal sensitivity
        self.prior_strength = prior_strength
        self.obs_strength = obs_strength
        # Probability this agent participates in any given round (1.0 = always)
        self.participation_rate = participation_rate

    def get_optimal_trade(self, market_price):
        """
        Calculates optimal trade size x* based on Eq 8 in Sethi et al. (2024).

        Returns:
            x_star (float): Number of shares to buy (+) or sell (-).
        """
        return compute_optimal_trade(
            belief=self.belief,
            price=market_price,
            cash=self.cash,
            shares=self.shares,
            rho=self.rho,
        )

    def update_portfolio(self, trade_shares, trade_cost):
        """
        Updates cash and shares after a trade is executed.

        Args:
            trade_shares (float): Number of shares bought (+) or sold (-).
            trade_cost (float): Total cost of the trade from the market maker.
        """
        self.cash -= trade_cost
        self.shares += trade_shares

    def update_belief(self, signal_s, *, method="beta", w=0.10, prior_strength=None, obs_strength=None):
        """
        Phase 2: update internal belief using public noisy signal S_t.

        method:
          - "weighted": p <- (1-w)p + w*S_t
          - "beta": Beta pseudo-count update (stable + interpretable)
        """
        from phase2_utils import update_belief_weighted, update_belief_beta

        # Fall back to per-agent values when caller doesn't override
        ps = self.prior_strength if prior_strength is None else float(prior_strength)
        os_ = self.obs_strength if obs_strength is None else float(obs_strength)

        if method == "weighted":
            self.belief = update_belief_weighted(self.belief, float(signal_s), float(w))
        elif method == "beta":
            self.belief = update_belief_beta(
                self.belief, float(signal_s),
                prior_strength=float(ps),
                obs_strength=float(os_),
            )
        else:
            raise ValueError(f"Unknown method={method!r}")
