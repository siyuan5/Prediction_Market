import sys
from pathlib import Path

# Ensure src directory is on sys.path to allow test imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.team_b_phase1_simulation import analyze_team_b_rho_effect, run_team_b_phase1


def test_phase1_generates_trades_and_price_path():
    # Run the simulation to ensure trades are executed and prices are updated over time
    result = run_team_b_phase1(
        seed=7,
        n_agents=50,
        n_rounds=150,
        fixed_rho=4.0,
        convergence_tol=0.01,
        stable_rounds=3,
        order_policy="hybrid",
    )
    assert len(result["price_series"]) >= 2  # At least two price points (start + some trading activity)
    assert sum(result["trade_count"]) > 0    # Ensure trades actually occurred
    assert sum(result["trade_volume"]) > 0   # There was meaningful trading volume


def test_higher_rho_reduces_position_size():
    # Analyze how position sizes change for different risk aversion (rho) values
    rows = analyze_team_b_rho_effect(
        rho_values=[0.5, 1.0, 2.0, 4.0],
        n_seeds=10,
        n_agents=40,
        n_rounds=250,
        convergence_tol=0.005,
        stable_rounds=3,
        order_policy="hybrid",
    )
    # Build mapping: rho value -> mean absolute position size
    by_rho = {row["rho"]: row["mean_abs_position"] for row in rows}
    # Higher rho (higher risk aversion) should always yield strictly smaller positions
    assert by_rho[1.0] < by_rho[0.5]
    assert by_rho[2.0] < by_rho[1.0]
    assert by_rho[4.0] < by_rho[2.0]


if __name__ == "__main__":
    # Run tests manually if this script is called directly
    test_phase1_generates_trades_and_price_path()
    test_higher_rho_reduces_position_size()
    print("Success")
