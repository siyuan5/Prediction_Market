import sys
from pathlib import Path

# Add project root to sys.path so test can import from src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.team_b_market_logic import ContinuousDoubleAuction

def test_cda_matches_crossing_orders():
    # Create the auction with an initial reference price
    exchange = ContinuousDoubleAuction(initial_reference_price=0.5)
    # Place a buy limit order; wants to buy up to 5 at price ≤ 0.60
    exchange.submit_limit_order(agent_id=1, side="buy", quantity=5.0, limit_price=0.60)
    # Place a sell limit order with a compatible (crossing) price; match expected 
    result = exchange.submit_limit_order(
        agent_id=2, side="sell", quantity=2.0, limit_price=0.55
    )
    trades = result["trades"]

    # Should execute one trade since buy price (0.60) ≥ sell price (0.55)
    assert len(trades) == 1
    assert trades[0].buyer_id == 1
    assert trades[0].seller_id == 2
    assert abs(trades[0].quantity - 2.0) < 1e-9  # Trade the full sell quantity
    assert abs(trades[0].price - 0.60) < 1e-9    # Price priority: executes at the buy order's price

if __name__ == "__main__":
    test_cda_matches_crossing_orders()
    print("Success")
