from .team_b_phase1_simulation import analyze_team_b_rho_effect, run_team_b_phase1


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


def main():
    results = run_team_b_phase1(
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


if __name__ == "__main__":
    main()
