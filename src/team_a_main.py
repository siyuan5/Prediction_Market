from team_a_phase1_simulation import run_phase1

def run_test(**kwargs):
    results = run_phase1(**kwargs)
    print("Phase 1 complete")
    print(f"Mean initial belief: {results['mean_initial_belief']:.4f}")
    print(f"Final price: {results['final_price']:.4f}")
    print(f"Final error vs P*: {results['final_error']:.4f}")
    print("Avg positions by rho:")
    for rho, summary in sorted(results["rho_summary"].items()):
        avg_shares = summary["avg_shares"]
        avg_cash = summary["avg_cash"]
        print(f"  rho={rho}: avg_shares={avg_shares:.4f}, avg_cash={avg_cash:.2f}")

def main():
    run_test()
    run_test(b=500)
    run_test(b=500)
    run_test(b=1000)
    run_test(initial_cash=250)
    run_test(rho_values=[0.75, 1.0, 1.25, 1.5, 2.0])
    run_test(b=4000, rho_values=[0.75, 1.0, 1.25, 1.5, 2.0])
    run_test(b=3500, rho_values=[0.75, 1.0, 1.25, 1.5, 2.0])

if __name__ == "__main__":
    main()
