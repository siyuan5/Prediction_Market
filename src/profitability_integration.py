"""
Integration guide: Adding profitability tracking to SimulationEngine.

This module shows how to wrap an existing SimulationEngine run with
profitability tracking and post-hoc analysis.

Key patterns:
1. Create ProfitabilitySession at simulation start
2. Register agents + record trades in real-time
3. Capture snapshots after each round
4. Export + visualize after simulation completes
"""

from typing import Any, Dict, Optional

try:
    from .simulation_engine import SimulationEngine
    from .profitability_analysis import ProfitabilitySession
    from .profitability_export import export_profitability_session, export_profitability_summary
    from .profitability_viz import ProfitabilityVisualizer
except ImportError:
    from simulation_engine import SimulationEngine
    from profitability_analysis import ProfitabilitySession
    from profitability_export import export_profitability_session, export_profitability_summary
    from profitability_viz import ProfitabilityVisualizer


class SimulationWithProfitability:
    """
    Wrapper around SimulationEngine that adds real-time profitability tracking.
    
    Usage:
        wrapper = SimulationWithProfitability(
            mechanism="lmsr",
            phase=1,
            n_agents=50,
            initial_cash=100.0,
        )
        wrapper.run_and_analyze(
            num_rounds=100,
            output_dir="outputs",
            run_name="my_experiment",
            generate_plots=True,
        )
    """
    
    def __init__(
        self,
        simulation_id: str = "default",
        **engine_kwargs: Any,
    ):
        """
        Initialize wrapper with SimulationEngine configuration.
        
        Args:
            simulation_id: identifier for this simulation run
            **engine_kwargs: all arguments to pass to SimulationEngine.__init__
        """
        self.simulation_id = simulation_id
        self.engine = SimulationEngine(**engine_kwargs)
        self.profitability = ProfitabilitySession(simulation_id=simulation_id)
        
        # Register agents with profitability tracker
        self._register_agents()
    
    def _register_agents(self) -> None:
        """Register all agents with profitability tracking."""
        for agent in self.engine.agents:
            self.profitability.register_agent(
                agent_id=agent.id,
                initial_cash=agent.cash,
                belief=agent.belief,
                rho=agent.rho,
            )
    
    def run_and_analyze(
        self,
        num_rounds: int,
        output_dir: str = "outputs",
        run_name: str = "profitability_run",
        generate_plots: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run simulation with profitability tracking and full post-hoc analysis.
        
        Args:
            num_rounds: number of simulation rounds
            output_dir: directory for outputs
            run_name: prefix for output files
            generate_plots: whether to create visualizations
            verbose: print progress updates
            
        Returns:
            Dictionary with paths to all generated artifacts
        """
        artifacts = {}
        
        if verbose:
            print(f"Running simulation ({num_rounds} rounds)...")
        
        # Run simulation, capturing profitability each round
        for round_num in range(num_rounds):
            if verbose and (round_num + 1) % max(1, num_rounds // 10) == 0:
                print(f"  Round {round_num + 1}/{num_rounds}")
            
            # Simulate one round
            self.engine.run(1)  # Run 1 more round
            
            # Get current state from engine
            # For both LMSR and CDA, access price through engine's price_series
            market_price = self.engine.price_series[-1] if self.engine.price_series else 0.5
            total_volume = self.engine.trade_volume[-1] if self.engine.trade_volume else 0.0
            current_signal = self.engine.signal_series[-1] if self.engine.signal_series else None
            
            # Update agent state from engine (capture cash/shares changes)
            for agent in self.engine.agents:
                self.profitability.agent_trackers[agent.id].cash = agent.cash
                self.profitability.agent_trackers[agent.id].shares = agent.shares
                self.profitability.agent_trackers[agent.id].last_price = market_price
                if agent.belief != self.profitability.agent_trackers[agent.id].belief:
                    self.profitability.update_agent_belief(agent.id, agent.belief)
            
            # Snapshot the round
            round_snapshot = self.profitability.snapshot_round(
                round_num=round_num,
                market_price=market_price,
                ground_truth=self.engine.ground_truth,
                total_volume=total_volume,
                signal=current_signal,
            )
            
            if verbose and round_num == 0:
                print(f"    Market price: {market_price:.6f}")
                print(f"    Avg profit: ${round_snapshot.avg_profit:.2f}")
        
        if verbose:
            print(f"Simulation complete. Generating analysis...\n")
        
        # Export data
        if verbose:
            print(f"Exporting profitability data...")
        export_artifacts = export_profitability_session(
            self.profitability,
            out_dir=output_dir,
            run_name=run_name,
        )
        artifacts.update(export_artifacts)
        
        # Export summary
        if verbose:
            print(f"Generating summary report...")
        summary_path = export_profitability_summary(
            self.profitability,
            out_dir=output_dir,
            run_name=run_name,
        )
        artifacts["summary"] = summary_path
        
        # Generate visualizations
        if generate_plots:
            if verbose:
                print(f"Generating visualizations...")
            try:
                visualizer = ProfitabilityVisualizer(self.profitability)
                plot_artifacts = visualizer.generate_all_plots(
                    output_dir=output_dir,
                    run_name=run_name,
                )
                artifacts.update(plot_artifacts)
                if verbose:
                    print(f"  Generated {len(plot_artifacts)} plots")
            except ImportError as e:
                if verbose:
                    print(f"  Skipping plots (matplotlib not available): {e}")
        
        if verbose:
            print(f"\nAnalysis complete!")
            print(f"Output directory: {output_dir}")
            print(f"Generated artifacts:")
            for label, path in artifacts.items():
                print(f"  - {label}: {path}")
        
        return artifacts
    
    def get_profitability_summary(self) -> Dict[str, Any]:
        """Get summary statistics from profitability session."""
        return self.profitability.get_summary()


# ============================================================================
# PATTERN: How to integrate with existing SimulationEngine runs
# ============================================================================

def add_profitability_to_existing_engine(
    engine: SimulationEngine,
    run_name: str = "analysis",
) -> ProfitabilitySession:
    """
    Attach profitability tracking to an already-initialized engine.
    
    Usage:
        engine = SimulationEngine(...)
        profitability = add_profitability_to_existing_engine(engine)
        engine.run(50)
        # ... periodically snapshot:
        #     profitability.snapshot_round(...)
    """
    profitability = ProfitabilitySession(simulation_id=run_name)
    
    for agent in engine.agents:
        profitability.register_agent(
            agent_id=agent.id,
            initial_cash=agent.cash,
            belief=agent.belief,
            rho=agent.rho,
        )
    
    return profitability


def snapshot_and_export(
    engine: SimulationEngine,
    profitability: ProfitabilitySession,
    output_dir: str = "outputs",
    run_name: str = "analysis",
) -> Dict[str, str]:
    """
    Snapshot current engine state into profitability and export all artifacts.
    
    NOTE: This function assumes the engine has already been run. It uses the
    engine's stored time series (price_series, trade_volume, signal_series) to
    generate profitability snapshots.
    
    Usage:
        engine = SimulationEngine(...)
        profitability = add_profitability_to_existing_engine(engine)
        
        for round_num in range(100):
            engine.run(1)
            # Profitability must be manually snapshotted if you want per-round data
        
        # Alternative: let this function reconstruct from engine history
        artifacts = snapshot_and_export(engine, profitability)
    """
    # If price_series exists, use it to backfill snapshots
    # (This assumes the profitability tracker was initialized empty)
    if engine.price_series and not profitability.round_snapshots:
        for round_num in range(len(engine.price_series)):
            market_price = engine.price_series[round_num]
            total_volume = engine.trade_volume[round_num] if round_num < len(engine.trade_volume) else 0.0
            signal = engine.signal_series[round_num] if round_num < len(engine.signal_series) else None
            
            # Update agent state from engine
            for agent in engine.agents:
                if agent.id in profitability.agent_trackers:
                    profitability.agent_trackers[agent.id].cash = agent.cash
                    profitability.agent_trackers[agent.id].shares = agent.shares
                    profitability.agent_trackers[agent.id].last_price = market_price
            
            # Snapshot
            profitability.snapshot_round(
                round_num=round_num,
                market_price=market_price,
                ground_truth=engine.ground_truth,
                total_volume=total_volume,
                signal=signal,
            )
    
    # Export
    artifacts = export_profitability_session(
        profitability,
        out_dir=output_dir,
        run_name=run_name,
    )
    
    # Summary
    summary_path = export_profitability_summary(
        profitability,
        out_dir=output_dir,
        run_name=run_name,
    )
    artifacts["summary"] = summary_path
    
    # Visualizations
    try:
        visualizer = ProfitabilityVisualizer(profitability)
        plot_artifacts = visualizer.generate_all_plots(
            output_dir=output_dir,
            run_name=run_name,
        )
        artifacts.update(plot_artifacts)
    except ImportError:
        pass  # matplotlib optional
    
    return artifacts
