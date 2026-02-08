import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.team_b_market_logic import ContinuousDoubleAuction


def test_cda_matches_crossing_orders():
    exchange = ContinuousDoubleAuction(initial_reference_price=0.5)
    exchange.submit_limit_order(agent_id=1, side="buy", quantity=5.0, limit_price=0.60)
    result = exchange.submit_limit_order(
        agent_id=2, side="sell", quantity=2.0, limit_price=0.55
    )
    trades = result["trades"]

    assert len(trades) == 1
    assert trades[0].buyer_id == 1
    assert trades[0].seller_id == 2
    assert abs(trades[0].quantity - 2.0) < 1e-9
    assert abs(trades[0].price - 0.60) < 1e-9


if __name__ == "__main__":
    test_cda_matches_crossing_orders()
    print("Success")
