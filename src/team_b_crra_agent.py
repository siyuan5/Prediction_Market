class TeamBCRRAAgent:
    def __init__(self, agent_id, initial_cash, belief_p, rho):
        self.id = int(agent_id)
        self.cash = float(initial_cash)
        self.shares = 0.0
        self.belief = float(belief_p)
        self.rho = float(rho)

    def get_optimal_trade(self, market_price):
        # Computes optimal risky asset trade for a CRRA agent at price `q`
        q = float(market_price)
        p = self.belief
        y = self.cash
        z = self.shares
        rho = self.rho

        # Out-of-bounds or indifference (skip degenerate and no-op scenarios)
        if q <= 0.01 or q >= 0.99:
            return 0.0
        if abs(p - q) < 1e-6:
            return 0.0

        numerator = p * (1 - q)
        denominator = q * (1 - p)
        # Avoid math errors and make sure trade direction is valid
        if numerator <= 0 or denominator <= 0:
            return 0.0

        # Closed-form for CRRA optimal trade under cash/holdings constraints
        k = (numerator / denominator) ** (1 / rho)
        return float(((k - 1) * y - z) / (1 + q * (k - 1)))

    def _max_buy_quantity(self, price):
        # Max shares that could be bought given all cash at given price
        p = max(float(price), 1e-9)
        return max(self.cash, 0.0) / p

    def _max_sell_quantity(self, price):
        # Max shares that could be sold, assuming portfolio is liquidated
        p = float(price)
        denom = max(1.0 - p, 1e-9)
        collateral = max(self.cash + self.shares, 0.0)
        return collateral / denom

    def _clip_quantity_for_price(self, side, quantity, price):
        # Ensures quantity does not breach buy/sell/collateral constraints
        q = max(float(quantity), 0.0)
        if side == "buy":
            return min(q, self._max_buy_quantity(price))
        return min(q, self._max_sell_quantity(price))

    def build_order(
        self,
        *,
        reference_price,
        best_bid=None,
        best_ask=None,
        order_policy="hybrid",
        limit_offset=0.01,
        market_order_edge=0.08,
        min_trade_size=1e-6,
    ):
        # Computes desired net position at reference price from CRRA optimality
        x_star = self.get_optimal_trade(reference_price)
        if abs(x_star) < min_trade_size:
            return None  # Ignore tiny or zero trades

        side = "buy" if x_star > 0 else "sell"
        target_qty = abs(x_star)

        use_market = False
        if order_policy == "market":
            use_market = True
        elif order_policy == "hybrid":
            # Use market order if belief is well outside spread and edge is large
            edge = abs(self.belief - reference_price)
            if side == "buy" and best_ask is not None:
                use_market = (self.belief >= best_ask) and (edge >= market_order_edge)
            elif side == "sell" and best_bid is not None:
                use_market = (self.belief <= best_bid) and (edge >= market_order_edge)

        if use_market:
            # Aggress market at extreme price (simulate certain fill)
            risk_price = 1.0 if side == "buy" else 0.0
            quantity = self._clip_quantity_for_price(side, target_qty, risk_price)
            if quantity < min_trade_size:
                return None
            return {
                "type": "market",
                "side": side,
                "quantity": quantity,
                "limit_price": None,
            }

        # Set price more favorable than belief to provide liquidity, bounded well inside (0, 1)
        if side == "buy":
            limit_price = min(max(self.belief - limit_offset, 0.001), 0.999)
        else:
            limit_price = min(max(self.belief + limit_offset, 0.001), 0.999)

        quantity = self._clip_quantity_for_price(side, target_qty, limit_price)
        if quantity < min_trade_size:
            return None

        return {
            "type": "limit",
            "side": side,
            "quantity": quantity,
            "limit_price": limit_price,
        }

    def update_portfolio(self, trade_shares, trade_cost):
        # Adjusts cash and shares to reflect an executed trade
        self.cash -= float(trade_cost)
        self.shares += float(trade_shares)

    def update_belief(self, signal_s, *, method="beta", w=0.10, prior_strength=20.0, obs_strength=5.0):
        # Bayesian or weighted belief update after observing a signal
        try:
            from .phase2_utils import update_belief_weighted, update_belief_beta
        except ImportError:
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
