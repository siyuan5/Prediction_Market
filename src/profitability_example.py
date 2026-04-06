"""
Example: Complete profitability analysis workflow.

This script demonstrates how to:
1. Run a simulation with profitability tracking
2. Export results to CSV/JSON
3. Generate visualizations
4. Analyze results

Run with: python profitability_example.py
"""

import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from simulation_engine import SimulationEngine
from profitability_integration import SimulationWithProfitability, add_profitability_to_existing_engine, snapshot_and_export
from profitability_export import export_profitability_session, export_profitability_summary
from profitability_viz import ProfitabilityVisualizer


def example_basic_run():
    """
    EXAMPLE 1: Basic workflow with profitability tracking.
    
    This is the simplest pattern: wrap your simulation with profitability.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Profitability Analysis")
    print("=" * 80 + "\n")
    
    # Create wrapper that adds profitability tracking
    wrapper = SimulationWithProfitability(
        simulation_id="example_run_1",
        mechanism="lmsr",
        phase=1,
        seed=42,
        ground_truth=0.70,
        n_agents=20,
        initial_cash=100.0,
        b=100.0,  # LMSR parameter
    )
    
    # Run simulation with full analysis
    artifacts = wrapper.run_and_analyze(
        num_rounds=50,
        output_dir="outputs/profitability_examples",
        run_name="example_basic",
        generate_plots=True,
        verbose=True,
    )
    
    # Print summary
    summary = wrapper.get_profitability_summary()
    print("\n" + "-" * 80)
    print("PROFITABILITY SUMMARY")
    print("-" * 80)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return artifacts


def example_manual_integration():
    """
    EXAMPLE 2: Manual integration with existing SimulationEngine.
    
    Use this pattern if you already have a SimulationEngine and want to
    add profitability tracking to existing code.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Manual Integration with Existing Engine")
    print("=" * 80 + "\n")
    
    # Create engine as you normally would
    engine = SimulationEngine(
        mechanism="lmsr",
        phase=1,
        seed=43,
        ground_truth=0.65,
        n_agents=30,
        initial_cash=100.0,
        b=100.0,
    )
    
    # Attach profitability tracking
    profitability = add_profitability_to_existing_engine(engine, run_name="example_manual")
    
    print(f"Running simulation with manual profitability tracking...")
    
    # Run for multiple rounds, snapshotting profitability each time
    for round_num in range(50):
        engine.run(1)  # Run 1 more round
        
        # Capture profitability snapshot
        market_price = engine.market.get_price() if hasattr(engine, 'market') else 0.5
        
        profitability.snapshot_round(
            round_num=round_num,
            market_price=market_price,
            ground_truth=engine.ground_truth,
            total_volume=0.0,  # You can track this from engine internals
            signal=None,
        )
        
        if (round_num + 1) % 10 == 0:
            print(f"  Round {round_num + 1}/50")
    
    # Export and analyze
    print("\nExporting and analyzing...")
    artifacts = snapshot_and_export(
        engine,
        profitability,
        output_dir="outputs/profitability_examples",
        run_name="example_manual",
    )
    
    print(f"Generated {len(artifacts)} artifacts")
    return artifacts


def example_phase2_with_signals():
    """
    EXAMPLE 3: Phase 2 simulation with belief updates.
    
    Demonstrates profitability tracking with signal-driven belief updates.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Phase 2 with Public Signals")
    print("=" * 80 + "\n")
    
    from phase2_utils import SignalSpec
    
    wrapper = SimulationWithProfitability(
        simulation_id="example_phase2",
        mechanism="lmsr",
        phase=2,
        seed=44,
        ground_truth=0.70,
        n_agents=25,
        initial_cash=100.0,
        b=100.0,
        signal_spec=SignalSpec(
            mode="binomial",  # 'bernoulli', 'binomial', or 'gaussian'
            n=25,  # number of samples for binomial mode
        ),
        belief_update_method="beta",
        belief_weight=0.10,
    )
    
    print("Running Phase 2 simulation with signals...")
    
    artifacts = wrapper.run_and_analyze(
        num_rounds=50,
        output_dir="outputs/profitability_examples",
        run_name="example_phase2",
        generate_plots=True,
        verbose=True,
    )
    
    print("\nPhase 2 analysis complete!")
    return artifacts


def example_comparative_analysis():
    """
    EXAMPLE 4: Compare profitability across multiple mechanisms.
    
    Run LMSR vs CDA and compare profit distributions.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Comparative Analysis (LMSR vs CDA)")
    print("=" * 80 + "\n")
    
    results = {}
    
    for mechanism in ["lmsr", "cda"]:
        print(f"\nRunning {mechanism.upper()} simulation...")
        
        try:
            # Get mechanism-specific parameters
            kwargs = {
                "simulation_id": f"comparative_{mechanism}",
                "mechanism": mechanism,
                "phase": 1,
                "seed": 45,
                "ground_truth": 0.70,
                "n_agents": 20,
                "initial_cash": 100.0,
            }
            
            # LMSR needs 'b' parameter; CDA needs 'initial_price'
            if mechanism == "lmsr":
                kwargs["b"] = 100.0
            else:
                kwargs["initial_price"] = 0.5
            
            wrapper = SimulationWithProfitability(**kwargs)
            
            artifacts = wrapper.run_and_analyze(
                num_rounds=40,
                output_dir="outputs/profitability_examples",
                run_name=f"example_comparison_{mechanism}",
                generate_plots=False,  # Skip plots for now
                verbose=False,
            )
            
            summary = wrapper.get_profitability_summary()
            results[mechanism] = summary
            
            print(f"  [OK] {mechanism.upper()}")
            print(f"    Final avg profit: ${summary['final_avg_profit']:.2f}")
            print(f"    Final Gini: {summary['final_gini']:.4f}")
            
        except Exception as e:
            print(f"  [ERROR] {mechanism.upper()}: {e}")
    
    # Comparison summary
    print("\n" + "-" * 80)
    print("COMPARISON SUMMARY")
    print("-" * 80)
    
    if len(results) == 2:
        lmsr_profit = results.get('lmsr', {}).get('final_avg_profit', 0)
        cda_profit = results.get('cda', {}).get('final_avg_profit', 0)
        
        print(f"LMSR avg profit:  ${lmsr_profit:.2f}")
        print(f"CDA avg profit:   ${cda_profit:.2f}")
        print(f"Difference:       ${abs(lmsr_profit - cda_profit):.2f}")
    
    return results


def example_detailed_analysis():
    """
    EXAMPLE 5: Detailed analysis of a single run.
    
    Shows how to generate all possible visualizations and summaries.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Detailed Single-Run Analysis")
    print("=" * 80 + "\n")
    
    wrapper = SimulationWithProfitability(
        simulation_id="detailed_analysis",
        mechanism="lmsr",
        phase=1,
        seed=46,
        ground_truth=0.70,
        n_agents=40,  # More agents for better distributions
        initial_cash=100.0,
        b=100.0,
    )
    
    print("Running simulation...")
    artifacts = wrapper.run_and_analyze(
        num_rounds=50,
        output_dir="outputs/profitability_examples",
        run_name="detailed_analysis",
        generate_plots=True,
        verbose=True,
    )
    
    # Print detailed summary
    profitability = wrapper.profitability
    
    if profitability.round_snapshots:
        final_round = profitability.round_snapshots[-1]
        
        print("\n" + "-" * 80)
        print("DETAILED FINAL ROUND ANALYSIS")
        print("-" * 80)
        
        print(f"\nMarket Price: {final_round.market_price:.6f}")
        print(f"Ground Truth: {final_round.ground_truth:.6f}")
        print(f"Price Error: {abs(final_round.market_price - final_round.ground_truth):.6f}")
        
        print(f"\n--- Profit Distribution ---")
        print(f"Mean:   ${final_round.avg_profit:.2f}")
        print(f"Median: ${final_round.median_profit:.2f}")
        print(f"Std:    ${final_round.std_profit:.2f}")
        print(f"Min:    ${final_round.min_profit:.2f}")
        print(f"Max:    ${final_round.max_profit:.2f}")
        print(f"Range:  ${final_round.max_profit - final_round.min_profit:.2f}")
        
        print(f"\n--- Inequality Metrics ---")
        print(f"Gini Coefficient: {final_round.gini_coefficient:.4f} (0=equal, 1=unequal)")
        
        print(f"\n--- Belief Accuracy ---")
        print(f"Avg Error: {final_round.avg_belief_error:.6f}")
        print(f"Accuracy Premium: {final_round.belief_accuracy_premium:.4f}")
        
        # Top/bottom performers
        sorted_agents = sorted(final_round.agent_snapshots, key=lambda a: a.total_pnl, reverse=True)
        
        print(f"\n--- Top 5 Agents ---")
        for i, agent in enumerate(sorted_agents[:5], 1):
            print(f"  {i}. Agent {agent.agent_id}: ${agent.total_pnl:.2f} (ρ={agent.rho:.2f})")
        
        print(f"\n--- Bottom 5 Agents ---")
        for i, agent in enumerate(reversed(sorted_agents[-5:]), 1):
            print(f"  {i}. Agent {agent.agent_id}: ${agent.total_pnl:.2f} (ρ={agent.rho:.2f})")
    
    return artifacts


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "PROFITABILITY ANALYSIS EXAMPLES")
    print("=" * 80)
    print()
    
    try:
        example_basic_run()
    except Exception as e:
        print(f"Error in Example 1: {e}\n")
    
    try:
        example_manual_integration()
    except Exception as e:
        print(f"Error in Example 2: {e}\n")
    
    try:
        example_phase2_with_signals()
    except Exception as e:
        print(f"Error in Example 3: {e}\n")
    
    try:
        example_comparative_analysis()
    except Exception as e:
        print(f"Error in Example 4: {e}\n")
    
    try:
        example_detailed_analysis()
    except Exception as e:
        print(f"Error in Example 5: {e}\n")
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nCheck the 'outputs/profitability_examples/' directory for results.")
    print()


if __name__ == "__main__":
    main()
