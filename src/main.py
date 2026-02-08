from phase1_simulation import run_phase1


def main():
    results = run_phase1()
    print("Phase 1 complete")
    print(f"Mean initial belief: {results['mean_initial_belief']:.4f}")
    print(f"Final price: {results['price_series'][-1]:.4f}")

if __name__ == "__main__":
    main()
