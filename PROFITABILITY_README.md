"""
PROFITABILITY ANALYSIS SYSTEM

A comprehensive framework for real-time and post-hoc profitability tracking
in prediction market simulations (LMSR and CDA).

================================================================================
OVERVIEW
================================================================================

The profitability system provides:

Real-time tracking of agent wealth, P&L, and portfolio metrics during simulation
Intra-round profitability snapshots (not just end-of-simulation)
Market-level aggregates: inequality metrics, belief-accuracy premiums, etc.
Multiple export formats: CSV, JSON, and human-readable summaries
Comprehensive visualizations: timeseries, distributions, scatter plots, etc.
Works with both LMSR (Team A) and CDA (Team B) mechanisms

Components:
  - profitability_analysis.py     : Core tracking & snapshot logic
  - profitability_export.py       : CSV/JSON export utilities
  - profitability_viz.py          : Matplotlib/seaborn visualizations
  - profitability_integration.py  : Integration with SimulationEngine
  - profitability_example.py      : Runnable examples and patterns

================================================================================
QUICK START
================================================================================

# Pattern 1: Simplest - Wrap your simulation
from profitability_integration import SimulationWithProfitability

wrapper = SimulationWithProfitability(
    mechanism="lmsr",
    phase=1,
    n_agents=50,
    initial_cash=100.0,
)

artifacts = wrapper.run_and_analyze(
    num_rounds=100,
    output_dir="outputs",
    run_name="my_experiment",
    generate_plots=True,
)


# Pattern 2: Integrate with existing SimulationEngine
from profitability_integration import add_profitability_to_existing_engine

engine = SimulationEngine(...)
profitability = add_profitability_to_existing_engine(engine)

for round_num in range(100):
    engine.run(1)
    profitability.snapshot_round(
        round_num=round_num,
        market_price=engine.market.get_price(),
        ground_truth=engine.ground_truth,
    )

# Export results
from profitability_integration import snapshot_and_export
artifacts = snapshot_and_export(engine, profitability)


================================================================================
ARCHITECTURE
================================================================================

Three-layer design:

LAYER 1: Real-time Tracking (profitability_analysis.py)
  - AgentProfitabilityTracker: tracks individual agent state
    * Portfolio: cash, shares, entry price
    * Trade history: all executed trades with prices
    * Metrics: realized P&L, unrealized P&L, total wealth
  
  - MarketProfitabilityAnalyzer: aggregates agent data
    * Profit statistics: mean, median, std, min, max
    * Inequality metrics: Gini coefficient
    * Belief-accuracy premium: correlation between belief accuracy and profit
  
  - ProfitabilitySession: manages entire simulation
    * Accumulates round-by-round snapshots
    * Aggregates agent state
    * Exports to dict format for further processing

LAYER 2: Export (profitability_export.py)
  - Multiple CSV formats:
    * profit_timeseries.csv: market-level metrics per round
    * agent_timeseries.csv: per-agent metrics per round
    * final_profits.csv: final ranking and profitability
    * inequality.csv: inequality metrics per round
  
  - JSON export: complete data structure for programmatic access
  
  - Human-readable summary: text report with key metrics

LAYER 3: Visualization (profitability_viz.py)
  - ProfitabilityVisualizer class
    * plot_profit_timeseries(): individual or aggregate trends
    * plot_belief_accuracy_vs_profit(): scatter with rho coloring
    * plot_rho_profit_distribution(): box/violin plots by risk aversion
    * plot_inequality_metrics(): Gini, std dev, range over time
    * plot_cumulative_volume_and_price(): market activity
    * plot_final_profit_distribution(): final profit histogram
    * generate_all_plots(): create all visualizations at once

================================================================================
FIXES APPLIED
================================================================================

The initial implementation had these issues that have been corrected:

1. **Agent state not being captured**: The integration now properly syncs agent
   cash and shares from the engine after each round. This ensures P&L is 
   accurately calculated from actual trading activity.

   Fix: After engine.run(1), we now sync agent state:
     for agent in engine.agents:
         profitability.agent_trackers[agent.id].cash = agent.cash
         profitability.agent_trackers[agent.id].shares = agent.shares

2. **Market price access compatibility**: LMSR and CDA have different APIs
   for getting the current price. Now handles both:
   - LMSR: engine.price_series[-1]
   - CDA: engine.price_series[-1] (via _current_price() method)

   Fix: Use engine.price_series which is populated by all mechanisms.

3. **SignalSpec parameter naming**: Phase 2 examples used wrong parameter names.
   
   Old (wrong): SignalSpec(signal_type="bernoulli", base_p=0.70, flip_prob=0.0)
   New (correct): SignalSpec(mode="binomial", n=25)

   Valid modes: "bernoulli", "binomial", "gaussian"

4. **Unicode encoding on Windows**: Examples used box-drawing characters that
   cause encoding errors. Now uses ASCII-safe alternatives.

================================================================================
KEY METRICS EXPLAINED
================================================================================

Per-Agent (per round):
  - cash: liquid capital available
  - shares: YES shares held
  - market_price: current market price
  - entry_price: weighted average acquisition price of shares
  - realized_pnl: profit from closed positions (always 0 in frictionless market)
  - unrealized_pnl: shares * (market_price - entry_price)
  - portfolio_value: cash + shares * market_price
  - total_pnl: portfolio_value - initial_cash

Market-level (per round):
  - avg_profit: mean total_pnl across agents
  - median_profit: median total_pnl
  - std_profit: standard deviation of profits
  - max_profit / min_profit: extremes
  - gini_coefficient: profit concentration (0=equal, 1=all to one agent)
  - avg_belief_error: mean |belief - ground_truth| across agents
  - belief_accuracy_premium: correlation between accurate beliefs and profits
    * Positive = accurate beliefs → higher profits (learning effect)
    * Negative = inaccurate beliefs → higher profits (lucky/contrarian)
    * Near zero = no correlation (random walk)

Inequality Metrics (inequality.csv):
  - profit_range: max_profit - min_profit
  - profit_std_dev: standard deviation
  - coefficient_of_variation: std / mean (0=uniform, high=dispersed)
  - top_10pct_profit: average profit of top 10% of agents
  - bottom_10pct_profit: average profit of bottom 10%

================================================================================
OUTPUT FILES
================================================================================

When you call run_and_analyze() or snapshot_and_export(), the system generates:

CSV FILES:
  {run_name}_profit_timeseries.csv
    - One row per round
    - Market price, profit statistics, volume, belief accuracy
    - Great for: understanding profit evolution over time

  {run_name}_agent_timeseries.csv
    - One row per agent per round
    - Each agent's cash, shares, portfolio value, P&L
    - Great for: individual agent trajectories, tracking specific agents

  {run_name}_final_profits.csv
    - One row per agent (sorted by profit)
    - Includes agent rank, percentile, rho value
    - Great for: final leaderboard, winner/loser analysis

  {run_name}_inequality.csv
    - One row per round
    - Gini coefficient, profit range, std dev, top/bottom 10% profits
    - Great for: understanding wealth concentration over time

JSON FILE:
  {run_name}_profitability.json
    - Complete nested structure with all snapshots and metrics
    - Suitable for: programmatic analysis, custom post-processing

TEXT SUMMARY:
  {run_name}_summary.txt
    - Human-readable markdown-style report
    - Top 5 and bottom 5 agents, key statistics
    - Suitable for: quick eyeballing of results

VISUALIZATIONS (PNG):
  {run_name}_profit_timeseries.png
    - Aggregate profit trend with confidence bands
  
  {run_name}_agent_timeseries.png
    - Individual agent profit lines (if ≤25 agents)
  
  {run_name}_belief_vs_profit.png
    - Scatter: belief accuracy vs profit, colored by rho
    - Shows whether accurate beliefs lead to profits
  
  {run_name}_rho_distribution.png
    - Box plot and violin plot of profit by risk aversion
    - Shows which rho values are most profitable
  
  {run_name}_inequality.png
    - 4 subplots: Gini, volatility, range, magnitude
    - Shows how equally/unequally profits are distributed
  
  {run_name}_volume_price.png
    - Market price vs ground truth + cumulative volume
    - Shows price discovery and liquidity
  
  {run_name}_final_distribution.png
    - Histogram of final profits
    - Shows profit distribution shape (normal, bimodal, etc.)

================================================================================
INTEGRATION PATTERNS
================================================================================

PATTERN 1: Simplest Wrapping
────────────────────────────────────
from profitability_integration import SimulationWithProfitability

wrapper = SimulationWithProfitability(
    simulation_id="my_run",
    mechanism="lmsr",
    phase=1,
    seed=42,
    n_agents=50,
    initial_cash=100.0,
    b=100.0,
)

artifacts = wrapper.run_and_analyze(
    num_rounds=100,
    output_dir="outputs",
    run_name="experiment1",
    generate_plots=True,
)


PATTERN 2: Manual Control
────────────────────────────────────
from profitability_integration import add_profitability_to_existing_engine
from simulation_engine import SimulationEngine

engine = SimulationEngine(
    mechanism="lmsr",
    phase=2,
    n_agents=30,
    ground_truth=0.70,
)

profitability = add_profitability_to_existing_engine(engine)

for round_num in range(100):
    engine.run(1)
    
    # Your custom logic here
    # ...
    
    # Snapshot profitability
    profitability.snapshot_round(
        round_num=round_num,
        market_price=engine.market.get_price(),
        ground_truth=engine.ground_truth,
        total_volume=0.0,  # Track if available
    )

artifacts = snapshot_and_export(engine, profitability)


PATTERN 3: Track Trades in Real-time
────────────────────────────────────
# In your trading logic:
profitability.record_trade(
    agent_id=agent.id,
    trade_qty=shares_to_buy,  # positive=buy, negative=sell
    trade_price=market_price,
    round_num=current_round,
)

# Later, when you snapshot:
profitability.snapshot_round(...)


PATTERN 4: Compare Multiple Scenarios
────────────────────────────────────
from profitability_integration import SimulationWithProfitability

scenarios = {
    "lmsr_phase1": dict(mechanism="lmsr", phase=1, b=100),
    "cda_phase1": dict(mechanism="cda", phase=1),
    "lmsr_phase2": dict(mechanism="lmsr", phase=2, b=100),
}

results = {}
for scenario_name, config in scenarios.items():
    wrapper = SimulationWithProfitability(
        simulation_id=scenario_name,
        **config,
        n_agents=50,
        initial_cash=100.0,
    )
    
    artifacts = wrapper.run_and_analyze(
        num_rounds=100,
        output_dir="outputs",
        run_name=scenario_name,
    )
    
    results[scenario_name] = wrapper.get_profitability_summary()


================================================================================
VISUALIZATION EXAMPLES
================================================================================

After running a simulation, generate visualizations:

from profitability_viz import ProfitabilityVisualizer

visualizer = ProfitabilityVisualizer(profitability_session)

# Individual plots
visualizer.plot_profit_timeseries(aggregate_only=True)
visualizer.plot_belief_accuracy_vs_profit(round_num=50)
visualizer.plot_rho_profit_distribution()

# Or generate all at once
plots = visualizer.generate_all_plots(
    output_dir="outputs",
    run_name="my_experiment",
)


================================================================================
API REFERENCE
================================================================================

See docstrings in the source files for detailed parameter descriptions.


profitability_analysis.py
─────────────────────────

class AgentProfitabilityTracker:
  __init__(agent_id, initial_cash, belief, rho)
  update_belief(new_belief)
  record_trade(trade_qty, trade_price, round_num)
  get_snapshot(round_num, market_price) -> AgentSnapshot
  get_trade_history() -> List[Dict]

class MarketProfitabilityAnalyzer:
  @staticmethod create_round_snapshot(...) -> RoundSnapshot

class ProfitabilitySession:
  __init__(simulation_id)
  register_agent(agent_id, initial_cash, belief, rho)
  update_agent_belief(agent_id, new_belief)
  record_trade(agent_id, trade_qty, trade_price, round_num)
  snapshot_round(round_num, market_price, ground_truth, ...) -> RoundSnapshot
  get_summary() -> Dict[str, Any]
  to_dict() -> Dict[str, Any]


profitability_export.py
──────────────────────

def export_profitability_session(
    session: ProfitabilitySession,
    out_dir: str,
    run_name: str,
) -> Dict[str, str]

def export_profitability_summary(
    session: ProfitabilitySession,
    out_dir: str,
    run_name: str,
) -> str


profitability_viz.py
───────────────────

class ProfitabilityVisualizer:
  __init__(session, style, figsize)
  plot_profit_timeseries(aggregate_only, output_path) -> Figure
  plot_belief_accuracy_vs_profit(round_num, output_path) -> Figure
  plot_rho_profit_distribution(round_num, output_path) -> Figure
  plot_inequality_metrics(output_path) -> Figure
  plot_cumulative_volume_and_price(output_path) -> Figure
  plot_final_profit_distribution(output_path) -> Figure
  generate_all_plots(output_dir, run_name) -> Dict[str, str]


profitability_integration.py
───────────────────────────

class SimulationWithProfitability:
  __init__(simulation_id, **engine_kwargs)
  run_and_analyze(num_rounds, output_dir, run_name, generate_plots, verbose)
  get_profitability_summary() -> Dict[str, Any]

def add_profitability_to_existing_engine(
    engine: SimulationEngine,
    run_name: str,
) -> ProfitabilitySession

def snapshot_and_export(
    engine: SimulationEngine,
    profitability: ProfitabilitySession,
    output_dir: str,
    run_name: str,
) -> Dict[str, str]


================================================================================
DEPENDENCIES
================================================================================

Required:
  - numpy (already in your project)

Optional:
  - matplotlib (for visualizations)
  - seaborn (for better plot styling)

Install optional dependencies with:
  pip install matplotlib seaborn

If matplotlib is not installed, you can still:
  - Use profitability_analysis.py (tracking)
  - Use profitability_export.py (CSV/JSON export)
  
But visualizations will be skipped with a helpful error message.


================================================================================
TROUBLESHOOTING
================================================================================

Q: "ModuleNotFoundError: No module named 'matplotlib'"
A: Run: pip install matplotlib seaborn
   Or skip visualization in code (just don't call generate_all_plots())

Q: Plots are not being generated
A: Check that ProfitabilityVisualizer was created without errors.
   Look for "matplotlib not available" messages.

Q: CSV files are empty
A: Make sure snapshot_round() was called after each round.
   If using manual integration, ensure profitability.snapshot_round(...) is in your loop.

Q: Profitability values look wrong
A: Verify that:
   1. initial_cash was set correctly in register_agent()
   2. record_trade() is being called with correct trade quantities
   3. market_price in snapshot_round() is current and correct

Q: How do I track trades from the CDA market?
A: In your CDA trading logic, call:
   profitability.record_trade(
       agent_id=trade.buyer_id,
       trade_qty=trade.quantity,
       trade_price=trade.price,
       round_num=current_round,
   )
   profitability.record_trade(
       agent_id=trade.seller_id,
       trade_qty=-trade.quantity,
       trade_price=trade.price,
       round_num=current_round,
   )


================================================================================
EXAMPLES
================================================================================

See profitability_example.py for 5 complete runnable examples:

1. Basic profitability analysis
2. Manual integration with existing engine
3. Phase 2 simulation with signals
4. Comparative analysis (LMSR vs CDA)
5. Detailed single-run analysis with custom metrics

Run with: python src/profitability_example.py


================================================================================
"""

# This module serves as documentation. For actual code, import from the modules listed above.
