import sys

# Ensure src directory is on Python path for local imports
sys.path.insert(0, "src")

from export_utils import export_phase2_results
from phase2_utils import SignalSpec
from team_b_phase2_simulation import run_team_b_phase2


def main():
    # Run the Team B Phase 2 simulation (continuous double auction with signals & belief updating)
    results = run_team_b_phase2(
        seed=42,                        # fixed seed for full reproducibility
        ground_truth=0.70,              # true event probability; used for scoring and agent learning
        n_agents=50,                    # number of agents in the market
        n_rounds=100,                   # time steps/rounds to run the market
        initial_cash=100.0,             # agent starting endowment
        sigma=0.10,                     # agent initial belief dispersion (standard deviation)
        rho_values=[0.75, 1.0, 1.25, 1.5, 2.0],  # heterogeneity in agent signal reliability
        signal_spec=SignalSpec(mode="binomial", n=25),  # per-round agent signals drawn from binomial dist.
        belief_update_method="beta",    # use Bayesian updating on beliefs after observing signals
        prior_strength=20.0,            # prior weight for agent Bayesian update
        obs_strength=5.0,               # weight given to each new signal observation
        shuffle_agents=True,            # randomize agent order each round (non-deterministic results)
        order_policy="hybrid",          # agents may mix between limit/market orders
        limit_offset=0.01,              # limit order price offset from agent value/belief
        market_order_edge=0.08,         # market order triggered if belief price exceeds edge threshold
    )

    # Export all result metrics and plots to disk for later analysis/visualization
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

