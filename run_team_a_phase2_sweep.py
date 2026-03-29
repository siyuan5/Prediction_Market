import os
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# Ensure the project src directory is on sys.path for local imports, regardless of working directory
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from phase2_utils import SignalSpec
from team_a_phase2_simulation import run_phase2


def main():
    # Grid search over agent heterogeneity (rho), signal noise (sigma), and market liquidity (b)
    rho_grids = [
        [0.75, 1.0, 1.25],          # moderate heterogeneity, centered around 1
        [0.5, 1.0, 2.0],            # more diverse agent signal reliability
    ]
    sigmas = [0.05, 0.10, 0.20]     # signal noise levels
    bs = [200.0, 500.0, 2000.0]     # LMSR liquidity parameter values
    seeds = [0, 1, 2, 3, 4]         # use multiple seeds for mean/stability

    out_path = "outputs/team_a_phase2_sweep_summary.csv"

    rows = []
    # Iterate over all parameter combinations
    for rho_values, sigma, b in product(rho_grids, sigmas, bs):
        final_prices = []
        final_errors = []
        # Repeat experiment for each random seed, aggregate results
        for seed in seeds:
            r = run_phase2(
                seed=seed,
                ground_truth=0.70,
                n_agents=50,
                n_rounds=60,
                initial_cash=100.0,
                sigma=sigma,
                rho_values=rho_values,
                b=b,
                signal_spec=SignalSpec(mode="binomial", n=25),
                belief_update_method="beta",
                prior_strength=20.0,
                obs_strength=5.0,
                shuffle_agents=False,  # deterministic agent order for reproducibility
            )
            final_prices.append(r["final_price"])
            final_errors.append(r["final_error"])

        # Compute mean results for this parameter setting
        avg_price = sum(final_prices) / len(final_prices)
        avg_err = sum(final_errors) / len(final_errors)

        rows.append({
            "rho_values": str(rho_values),  # store as string for CSV readability
            "sigma": sigma,
            "b": b,
            "seeds": str(seeds),
            "avg_final_price": avg_price,
            "avg_final_error": avg_err,
            "min_final_error": min(final_errors),
            "max_final_error": max(final_errors),
        })

    # Write sweep results to CSV so they can be easily inspected or plotted elsewhere
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sweep summary: {out_path}")
    output_graphs(rows)

def output_graphs(rows):
    # Plot experiment results across key parameter axes for visual analysis
    df = pd.DataFrame(rows)

    # Final error vs. liquidity for each sigma value
    for sigma in sorted(df["sigma"].unique()):
        subset = df[df["sigma"] == sigma]
        plt.plot(
            subset["b"],
            subset["avg_final_error"],
            marker="o",
            label=f"sigma={sigma}"
        )

    plt.xlabel("Liquidity Parameter b")
    plt.ylabel("Average Final Error")
    plt.title("Final Error vs Liquidity (b)")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/error_vs_b.png")
    plt.clf()

    # Final error vs. signal noise for each liquidity value
    for b in sorted(df["b"].unique()):
        subset = df[df["b"] == b]
        plt.plot(
            subset["sigma"],
            subset["avg_final_error"],
            marker="o",
            label=f"b={b}"
        )

    plt.xlabel("Initial Belief Dispersion (sigma)")
    plt.ylabel("Average Final Error")
    plt.title("Final Error vs Belief Dispersion")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/error_vs_sigma.png")
    plt.clf()

    # Final average price vs. liquidity for each sigma
    for sigma in sorted(df["sigma"].unique()):
        subset = df[df["sigma"] == sigma]
        plt.plot(
            subset["b"],
            subset["avg_final_price"],
            marker="o",
            label=f"sigma={sigma}"
        )

    plt.xlabel("Liquidity Parameter b")
    plt.ylabel("Average Final Price")
    plt.title("Final Price vs Liquidity (b)")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/price_vs_b.png")
    plt.clf()


if __name__ == "__main__":
    main()