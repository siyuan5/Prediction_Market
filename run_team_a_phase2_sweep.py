import os
import sys
import csv
from itertools import product

# Allow importing from /src when running from repo root
ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from phase2_utils import SignalSpec
from team_a_phase2_simulation import run_phase2


def main():
    # Simple sweep: vary rho grid + signal noise (sigma)
    rho_grids = [
        [0.75, 1.0, 1.25],          # baseline-ish
        [0.5, 1.0, 2.0],            # wider heterogeneity
    ]
    sigmas = [0.05, 0.10, 0.20]
    bs = [200.0, 500.0, 2000.0]
    seeds = [0, 1, 2, 3, 4]         # average over a few seeds

    out_path = "outputs/team_a_phase2_sweep_summary.csv"

    rows = []
    for rho_values, sigma, b in product(rho_grids, sigmas, bs):
        final_prices = []
        final_errors = []
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
                shuffle_agents=False,
            )
            final_prices.append(r["final_price"])
            final_errors.append(r["final_error"])

        avg_price = sum(final_prices) / len(final_prices)
        avg_err = sum(final_errors) / len(final_errors)

        rows.append({
            "rho_values": str(rho_values),
            "sigma": sigma,
            "b": b,
            "seeds": str(seeds),
            "avg_final_price": avg_price,
            "avg_final_error": avg_err,
            "min_final_error": min(final_errors),
            "max_final_error": max(final_errors),
        })

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sweep summary: {out_path}")


if __name__ == "__main__":
    main()