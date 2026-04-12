"""
Shared CRRA trade-sizing helper used by simulator and autonomous agents.
"""


def compute_optimal_trade(belief, price, cash, shares, rho):
    """
    Calculate the agent's optimal net YES-share trade size.

    Args:
        belief: Agent's subjective probability for the event.
        price: Current market YES price.
        cash: Agent cash available before the trade.
        shares: Agent's current YES-share holdings.
        rho: CRRA risk-aversion parameter.

    Returns:
        float: Shares to buy (+) or sell (-).
    """
    q = float(price)
    p = float(belief)
    y = float(cash)
    z = float(shares)
    rho = float(rho)

    # Avoid division by zero or unstable odds at extreme prices.
    if q <= 0.01 or q >= 0.99:
        return 0.0

    # If the agent agrees with the market, skip the trade for stability.
    if abs(p - q) < 1e-6:
        return 0.0

    numerator = p * (1 - q)
    denominator = q * (1 - p)
    if numerator <= 0 or denominator <= 0:
        return 0.0

    # Risk-adjusted edge from the CRRA closed-form solution.
    k = (numerator / denominator) ** (1 / rho)
    x_star = ((k - 1) * y - z) / (1 + q * (k - 1))

    # Clip buys by available cash and sells by simplified margin capacity.
    max_buy = y / q if q > 0 else 0.0
    if x_star > 0:
        return min(x_star, max_buy)

    max_sell = y / (1 - q) if (1 - q) > 0 else 0.0
    return max(x_star, -max_sell)