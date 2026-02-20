import sys

sys.path.insert(0, "src")

from export_utils import export_phase2_results
from phase2_utils import SignalSpec
from team_a_phase2_simulation import run_phase2


def main():
    results = run_phase2(
        seed=42,
        ground_truth=0.70,
        n_agents=50,
        n_rounds=100,
        initial_cash=100.0,
        sigma=0.10,
        rho_values=[0.75, 1.0, 1.25, 1.5, 2.0],
        b=4000.0,
        signal_spec=SignalSpec(mode="binomial", n=25),
        belief_update_method="beta",
        prior_strength=20.0,
        obs_strength=5.0,
        shuffle_agents=False,
    )

    paths = export_phase2_results(
        results,
        out_dir="outputs",
        run_name="team_a_phase2_baseline",
    )

    print("Phase 2 complete (Team A - LMSR)")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Final error vs P*: {results['final_error']:.4f}")
    print(f"Saved files: {paths}")


if __name__ == "__main__":
    main()
