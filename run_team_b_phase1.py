import sys
# Ensure 'src' directory is on the module search path for local imports
sys.path.insert(0, 'src')

from team_b_phase1_simulation import run_team_b_phase1, analyze_team_b_rho_effect

def main():
    # Run a baseline simulation for Team B's Phase 1 CDA market with typical parameters
    results = run_team_b_phase1(
        convergence_tol=0.01,     # Price considered converged if change is below this tolerance
        stable_rounds=5,          # Require this many rounds of stability to claim convergence
        order_policy="hybrid",    # Agents select limit/market order based on edge/offset rules
        limit_offset=0.01,        # Limit orders placed offset from fundamental value
        market_order_edge=0.08,   # Submit market order if belief price is beyond this threshold
    )
    # Track how different the market price is from the average initial agent belief
    price_gap = abs(results["final_price"] - results["mean_initial_belief"])

    print("Phase 1 complete (Team B - Continuous Double Auction)")
    print(f"Mean initial belief: {results['mean_initial_belief']:.4f}")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Absolute gap: {price_gap:.6f}")
    print(f"Converged (tol={results['convergence_tol']}): {results['converged']}")
    print(f"Rounds run: {results['rounds_run']}")
    print(f"Total executed volume: {sum(results['trade_volume']):.2f}")
    print(f"Total trades: {sum(results['trade_count'])}")
    
    print("\nRho Analysis:")
    # Evaluate how agent belief heterogeneity (rho) impacts learning and market outcomes
    rho_rows = analyze_team_b_rho_effect(
        rho_values=[0.5, 1.0, 2.0],  # Low/med/high heterogeneity in agent beliefs
        n_seeds=10,                  # Repeat experiments for robustness to randomness
        ground_truth=0.70,           # True event probability for simulation
        n_agents=50,
        n_rounds=500,
        initial_cash=100.0,
        sigma=0.10,
        convergence_tol=0.01,
        stable_rounds=5,
        order_policy="hybrid",
        limit_offset=0.01,
        market_order_edge=0.08,
    )
    
    print("\nRho sweep:")
    print("rho\tmean|position|\tmean|final-mean belief|\tconv_rate\tmean_rounds")
    for row in rho_rows:
        # Each row summarizes performance metrics for a given rho value averaged over seeds
        print(f"{row['rho']:.2f}\t{row['mean_abs_position']:.4f}\t\t{row['mean_final_price_gap']:.6f}\t\t{row['convergence_rate']:.2f}\t\t{row['mean_rounds']:.1f}")

if __name__ == "__main__":
    main()
