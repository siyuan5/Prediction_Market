import sys

sys.path.insert(0, "src")

from export_utils import export_phase2_results
from phase2_utils import SignalSpec
from team_b_phase2_simulation import run_team_b_phase2


def main():
    results = run_team_b_phase2(
        seed=42,
        ground_truth=0.70,
        n_agents=50,
        n_rounds=100,
        initial_cash=100.0,
        sigma=0.10,
        rho_values=[0.75, 1.0, 1.25, 1.5, 2.0],
        signal_spec=SignalSpec(mode="binomial", n=25),
        belief_update_method="beta",
        prior_strength=20.0,
        obs_strength=5.0,
        shuffle_agents=True,
        order_policy="hybrid",
        limit_offset=0.01,
        market_order_edge=0.08,
    )

    paths = export_phase2_results(
        results,
        out_dir="outputs",
        run_name="team_b_phase2_baseline",
    )

    print("Phase 2 complete (Team B - Continuous Double Auction)")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Final error vs P*: {results['final_error']:.4f}")
    print(f"Rounds run: {results['rounds_run']}")
    print(f"Total executed volume: {sum(results['trade_volume']):.2f}")
    print(f"Total trades: {sum(results['trade_count'])}")
    print(f"Saved files: {paths}")


if __name__ == "__main__":
    main()

