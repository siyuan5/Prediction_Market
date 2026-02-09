from team_a_phase1_simulation import run_phase1
from team_a_phase1_simulation import run_phase1
from export_utils import export_phase1_results

def run_test(run_name="phase1_run", **kwargs):
    results = run_phase1(**kwargs)

    # export JSON + CSVs
    paths = export_phase1_results(results, out_dir="outputs", run_name=run_name)
    print(f"Exported: {paths}")

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
    run_test(run_name="test1_baseline")
    run_test(run_name="test2_b500", b=500)
    run_test(run_name="test3_b500_repeat", b=500)
    run_test(run_name="test4_b1000", b=1000)
    run_test(run_name="test5_cash250", initial_cash=250)
    run_test(run_name="test6_rho_sweep", rho_values=[0.75, 1.0, 1.25, 1.5, 2.0])
    run_test(run_name="test7_b4000_rho_sweep", b=4000, rho_values=[0.75, 1.0, 1.25, 1.5, 2.0])
    run_test(run_name="test8_b3500_rho_sweep", b=3500, rho_values=[0.75, 1.0, 1.25, 1.5, 2.0])


if __name__ == "__main__":
    main()
