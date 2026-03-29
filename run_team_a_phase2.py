import sys

# Allow for importing from the local src directory in project layout
sys.path.insert(0, "src")

from export_utils import export_phase2_results
from phase2_utils import SignalSpec
from team_a_phase2_simulation import run_phase2


def main():
    # Run a baseline Phase 2 simulation with specified market and agent parameters
    results = run_phase2(
        seed=42,                       # fixed seed for reproducibility
        ground_truth=0.70,             # true event probability (P*) for scoring error
        n_agents=50,
        n_rounds=100,
        initial_cash=100.0,
        sigma=0.10,                    # initial agent belief dispersion
        rho_values=[0.75, 1.0, 1.25, 1.5, 2.0],  # range of agent reliability (heterogeneity) values
        b=500.0,                       # LMSR liquidity parameter
        signal_spec=SignalSpec(mode="binomial", n=25),   # per-round agent signals
        belief_update_method="beta",   # how agents update beliefs after signals
        prior_strength=20.0,           # prior strength for Bayesian updates
        obs_strength=10.0,             # strength/weight given to observed signals
        trade_fraction=0.20,           # fraction of cash available to trade per round
        shuffle_agents=False,          # deterministic agent order for full reproducibility
    )

    # Export all simulation results to disk: CSVs, plots, config metadata
    paths = export_phase2_results(
        results,
        out_dir="outputs",
        run_name="team_a_phase2_baseline",
    )

    # Print summary of simulation and location of saved files for user
    print("Phase 2 complete (Team A - LMSR)")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Final error vs P*: {results['final_error']:.4f}")
    print(f"Saved files: {paths}")


if __name__ == "__main__":
    main()
