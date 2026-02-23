import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_utils import SignalSpec
from team_b_phase1_simulation import (
    analyze_team_b_rho_effect,
    estimate_required_rounds_phase1,
    run_team_b_phase1,
)
from team_b_phase2_simulation import estimate_required_rounds_phase2, run_team_b_phase2


def _print_rho_table(rows):
    print("\nRho sweep (homogeneous populations, CDA order book):")
    print(
        "rho\tmean|position|\tstd|position|\tmean|final-mean belief|\tconv_rate\tmean_rounds\tmean_volume"
    )
    for row in rows:
        print(
            f"{row['rho']:.2f}\t{row['mean_abs_position']:.4f}\t\t"
            f"{row['std_abs_position']:.4f}\t\t{row['mean_final_price_gap']:.6f}\t\t"
            f"{row['convergence_rate']:.2f}\t\t{row['mean_rounds']:.1f}\t\t"
            f"{row['mean_total_volume']:.2f}"
        )


def _print_required_rounds():
    """Estimate required n_rounds for Phase 1 and Phase 2 from multi-seed runs."""
    common = dict(
        ground_truth=0.70,
        n_agents=50,
        initial_cash=100.0,
        sigma=0.10,
        order_policy="hybrid",
        limit_offset=0.01,
        market_order_edge=0.08,
    )
    print("Estimating required rounds for Team B (multiple seeds)...")
    p1 = estimate_required_rounds_phase1(
        n_seeds=30,
        percentile=95,
        n_rounds_cap=2000,
        convergence_tol=0.01,
        stable_rounds=5,
        **common,
    )
    print("\n--- Phase 1 (static beliefs, converge to mean belief) ---")
    print(f"  Mean rounds to converge:     {p1['mean_rounds']:.1f} ± {p1['std_rounds']:.1f}")
    print(f"  Min / max:                  {p1['min_rounds']} / {p1['max_rounds']}")
    print(f"  Convergence rate:          {p1['convergence_rate']:.2%}")
    print(f"  Recommended n_rounds (p{p1['percentile_used']}): {p1['recommended_n_rounds']}")

    p2_kw = {**common, "signal_spec": None, "belief_update_method": "beta"}
    p2 = estimate_required_rounds_phase2(
        n_seeds=20,
        n_rounds_cap=200,
        percentile=95,
        error_threshold=0.0008,
        **p2_kw,
    )
    print("\n--- Phase 2 (updating beliefs, price tracks P*) ---")
    print(f"  Error threshold:            |price - P*| <= {p2['error_threshold']}")
    print(f"  Runs that hit threshold:    {p2['runs_that_hit_threshold']}/{p2['n_seeds']}")
    print(f"  Mean rounds to threshold:   {p2['mean_rounds_to_threshold']:.1f}")
    print(f"  Mean final error:           {p2['mean_final_error']:.4f}")
    print(f"  Recommended n_rounds (p{p2['percentile_used']}): {p2['recommended_n_rounds']}")
    print()

def _run_phase1(seed=42):
    results = run_team_b_phase1(
        seed=seed,
        convergence_tol=0.01,
        stable_rounds=5,
        order_policy="hybrid",
        limit_offset=0.01,
        market_order_edge=0.08,
    )
    price_gap = abs(results["final_price"] - results["mean_initial_belief"])

    print("Phase 1 complete (Team B - Continuous Double Auction)")
    print(f"Mean initial belief: {results['mean_initial_belief']:.4f}")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Absolute gap: {price_gap:.6f}")
    print(f"Converged (tol={results['convergence_tol']}): {results['converged']}")
    print(f"Rounds run: {results['rounds_run']}")
    print(f"Total executed volume: {sum(results['trade_volume']):.2f}")
    print(f"Total trades: {sum(results['trade_count'])}")

    rho_rows = analyze_team_b_rho_effect(
        rho_values=[0.5, 1.0, 2.0, 4.0],
        n_seeds=30,
        ground_truth=0.70,
        n_agents=50,
        n_rounds=500,
        initial_cash=100.0,
        sigma=0.10,
        convergence_tol=0.01,
        stable_rounds=5,
        min_trade_size=1e-6,
        shuffle_agents=True,
        order_policy="hybrid",
        limit_offset=0.01,
        market_order_edge=0.08,
    )
    _print_rho_table(rho_rows)


def _run_phase2(seed=42, signal_mode="binomial", signal_n=25, signal_sigma=0.08, belief_method="beta"):
    """Run Phase 2: fixed P*, noisy signals S_t, belief updates; observe price tracking P*."""
    if signal_mode == "gaussian":
        signal_spec = SignalSpec(mode="gaussian", sigma=signal_sigma)
    elif signal_mode == "bernoulli":
        signal_spec = SignalSpec(mode="bernoulli")
    else:
        signal_spec = SignalSpec(mode="binomial", n=signal_n)

    results = run_team_b_phase2(
        seed=seed,
        ground_truth=0.70,
        n_agents=50,
        n_rounds=100,
        initial_cash=100.0,
        sigma=0.10,
        signal_spec=signal_spec,
        belief_update_method=belief_method,
        order_policy="hybrid",
        limit_offset=0.01,
        market_order_edge=0.08,
    )

    print("Phase 2 complete (Team B - Continuous Double Auction)")
    print(f"Ground truth P*: {results['ground_truth']:.4f}")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Final error |q - P*|: {results['final_error']:.4f}")
    print(f"Rounds run: {results['rounds_run']}")
    print(f"Total executed volume: {sum(results['trade_volume']):.2f}")
    print(f"Total trades: {sum(results['trade_count'])}")
    print(f"Signal mode: {results.get('signal_mode', signal_mode)}")
    print(f"Belief update: {results['belief_update_method']}")


def main():
    parser = argparse.ArgumentParser(
        description="Team B (CDA): Phase 1 (static beliefs) or Phase 2 (updating beliefs)."
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Scenario preset: 1 = static beliefs, 2 = noisy signals + belief updates (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--signal-mode",
        type=str,
        choices=["bernoulli", "binomial", "gaussian"],
        default="binomial",
        help="Phase 2 only: noisy signal type (default: binomial)",
    )
    parser.add_argument(
        "--signal-n",
        type=int,
        default=25,
        help="Phase 2 only: binomial sample size (noise level; default: 25)",
    )
    parser.add_argument(
        "--signal-sigma",
        type=float,
        default=0.08,
        help="Phase 2 only: gaussian signal std (noise level; default: 0.08)",
    )
    parser.add_argument(
        "--belief-method",
        type=str,
        choices=["weighted", "beta"],
        default="beta",
        help="Phase 2 only: belief update method (default: beta)",
    )
    parser.add_argument(
        "--estimate-rounds",
        action="store_true",
        help="Run Phase 1 and Phase 2 round estimators and print recommended n_rounds; then exit",
    )
    args = parser.parse_args()

    if args.estimate_rounds:
        _print_required_rounds()
        return

    if args.phase == 1:
        _run_phase1(seed=args.seed)
    else:
        _run_phase2(
            seed=args.seed,
            signal_mode=args.signal_mode,
            signal_n=args.signal_n,
            signal_sigma=args.signal_sigma,
            belief_method=args.belief_method,
        )


if __name__ == "__main__":
    main()
