import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase2_utils import SignalSpec
from src.team_b_phase2_simulation import run_team_b_phase2


def test_team_b_phase2_returns_tracking_series():
    result = run_team_b_phase2(
        seed=7,
        n_agents=40,
        n_rounds=20,
        fixed_rho=2.0,
        signal_spec=SignalSpec(mode="binomial", n=20),
        order_policy="hybrid",
    )

    assert result["rounds_run"] == 20
    assert len(result["price_series"]) == 20
    assert len(result["signal_series"]) == 20
    assert len(result["mean_belief_series"]) == 20
    assert len(result["error_series"]) == 20
    assert len(result["trade_volume"]) == 20
    assert len(result["trade_count"]) == 20
    assert len(result["final_beliefs"]) == 40
    assert all(0.01 <= p <= 0.99 for p in result["price_series"])
    assert all(0.01 <= b <= 0.99 for b in result["final_beliefs"])


def test_team_b_phase2_updates_beliefs_over_time():
    result = run_team_b_phase2(
        seed=3,
        n_agents=30,
        n_rounds=5,
        fixed_rho=1.0,
        signal_spec=SignalSpec(mode="bernoulli"),
        belief_update_method="weighted",
        belief_weight=0.5,
        order_policy="hybrid",
    )

    initial = result["initial_beliefs"]
    final = result["final_beliefs"]
    assert len(initial) == len(final) == 30
    assert any(abs(float(f) - float(i)) > 1e-9 for i, f in zip(initial, final))
    assert result["belief_update_method"] == "weighted"

