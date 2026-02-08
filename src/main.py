from phase1_simulation import run_phase1


def main():
    results = run_phase1()
    print("Phase 1 complete")
    print(f"Mean initial belief: {results['mean_initial_belief']:.4f}")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Final error vs P*: {results['final_error']:.4f}")
    print("Avg positions by rho:")
    for rho, summary in sorted(results["rho_summary"].items()):
        avg_shares = summary["avg_shares"]
        avg_cash = summary["avg_cash"]
        print(f"  rho={rho}: avg_shares={avg_shares:.4f}, avg_cash={avg_cash:.2f}")

if __name__ == "__main__":
    main()
