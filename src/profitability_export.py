"""
Export profitability analysis data to CSV and JSON formats.

Extends existing export_utils.py with profitability-specific exporters.
Handles both flat timeseries exports and detailed agent-level snapshots.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

try:
    from .profitability_analysis import ProfitabilitySession, RoundSnapshot
except ImportError:
    from profitability_analysis import ProfitabilitySession, RoundSnapshot


def _ensure_dir(path: Path) -> None:
    """Create directory (including parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def export_profitability_session(
    session: ProfitabilitySession,
    *,
    out_dir: str = "outputs",
    run_name: str = "profitability_run",
) -> Dict[str, str]:
    """
    Export full ProfitabilitySession data to JSON and multiple CSV files.
    
    Output files:
      - <run_name>_profitability.json:     Complete profitability data
      - <run_name>_profit_timeseries.csv:  Market-level profit metrics per round
      - <run_name>_agent_timeseries.csv:   Per-agent profit data (one row per agent per round)
      - <run_name>_final_profits.csv:      Final profit snapshot (one row per agent)
      - <run_name>_inequality.csv:         Market inequality metrics per round
    
    Returns:
        Mapping from artifact label to output file path
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)
    
    artifacts = {}
    
    # ---- Full data to JSON ----
    json_path = out_path / f"{run_name}_profitability.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2)
    artifacts["profitability_json"] = str(json_path)
    
    # ---- Market-level timeseries (one row per round) ----
    ts_path = out_path / f"{run_name}_profit_timeseries.csv"
    with ts_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "market_price",
            "ground_truth",
            "signal",
            "total_volume",
            "avg_profit",
            "median_profit",
            "std_profit",
            "max_profit",
            "min_profit",
            "profit_range",
            "avg_belief_error",
            "belief_accuracy_premium",
        ])
        
        for rs in session.round_snapshots:
            writer.writerow([
                rs.round_num,
                f"{rs.market_price:.6f}",
                f"{rs.ground_truth:.6f}",
                f"{rs.signal:.6f}" if rs.signal is not None else "",
                f"{rs.total_volume:.6f}",
                f"{rs.avg_profit:.6f}",
                f"{rs.median_profit:.6f}",
                f"{rs.std_profit:.6f}",
                f"{rs.max_profit:.6f}",
                f"{rs.min_profit:.6f}",
                f"{rs.max_profit - rs.min_profit:.6f}",
                f"{rs.avg_belief_error:.6f}",
                f"{rs.belief_accuracy_premium:.6f}",
            ])
    artifacts["profit_timeseries"] = str(ts_path)
    
    # ---- Per-agent timeseries (one row per agent per round) ----
    agent_ts_path = out_path / f"{run_name}_agent_timeseries.csv"
    with agent_ts_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "agent_id",
            "rho",
            "belief",
            "cash",
            "shares",
            "market_price",
            "entry_price",
            "realized_pnl",
            "unrealized_pnl",
            "portfolio_value",
            "total_pnl",
            "trades_this_round",
            "net_shares_traded",
        ])
        
        for rs in session.round_snapshots:
            for agent_snap in rs.agent_snapshots:
                writer.writerow([
                    rs.round_num,
                    agent_snap.agent_id,
                    f"{agent_snap.rho:.6f}",
                    f"{agent_snap.belief:.6f}",
                    f"{agent_snap.cash:.6f}",
                    f"{agent_snap.shares:.6f}",
                    f"{agent_snap.market_price:.6f}",
                    f"{agent_snap.entry_price:.6f}",
                    f"{agent_snap.realized_pnl:.6f}",
                    f"{agent_snap.unrealized_pnl:.6f}",
                    f"{agent_snap.portfolio_value:.6f}",
                    f"{agent_snap.total_pnl:.6f}",
                    agent_snap.trades_this_round,
                    f"{agent_snap.net_shares_traded:.6f}",
                ])
    artifacts["agent_timeseries"] = str(agent_ts_path)
    
    # ---- Final profits (one row per agent) ----
    if session.round_snapshots:
        final_round = session.round_snapshots[-1]
        final_path = out_path / f"{run_name}_final_profits.csv"
        with final_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "agent_id",
                "rho",
                "initial_belief",
                "final_belief",
                "final_cash",
                "final_shares",
                "final_portfolio_value",
                "total_pnl",
                "final_market_price",
                "entry_price",
                "unrealized_pnl_at_end",
                "profit_rank",
                "profit_percentile",
            ])
            
            # Sort by profit for ranking
            sorted_agents = sorted(
                final_round.agent_snapshots,
                key=lambda a: a.total_pnl,
                reverse=True
            )
            
            for rank, agent_snap in enumerate(sorted_agents, 1):
                percentile = (rank - 1) / len(sorted_agents) * 100 if sorted_agents else 0
                writer.writerow([
                    agent_snap.agent_id,
                    f"{agent_snap.rho:.6f}",
                    f"{session.agent_trackers[agent_snap.agent_id].belief:.6f}",
                    f"{agent_snap.belief:.6f}",
                    f"{agent_snap.cash:.6f}",
                    f"{agent_snap.shares:.6f}",
                    f"{agent_snap.portfolio_value:.6f}",
                    f"{agent_snap.total_pnl:.6f}",
                    f"{agent_snap.market_price:.6f}",
                    f"{agent_snap.entry_price:.6f}",
                    f"{agent_snap.unrealized_pnl:.6f}",
                    rank,
                    f"{percentile:.2f}",
                ])
        artifacts["final_profits"] = str(final_path)
    
    # ---- Inequality metrics ----
    ineq_path = out_path / f"{run_name}_inequality.csv"
    with ineq_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "round",
            "market_price",
            "gini_coefficient",
            "profit_range",
            "profit_std_dev",
            "coefficient_of_variation",
            "top_10pct_profit",
            "bottom_10pct_profit",
        ])
        
        for rs in session.round_snapshots:
            profits = [s.total_pnl for s in rs.agent_snapshots]
            
            if profits:
                sorted_profits = sorted(profits, reverse=True)
                n = len(sorted_profits)
                top_10_threshold = max(1, n // 10)
                bottom_10_threshold = -max(1, n // 10) if n > 1 else None
                
                top_10_profit = float(np.mean(sorted_profits[:top_10_threshold]))
                bottom_10_profit = float(np.mean(sorted_profits[bottom_10_threshold:]))
                
                cv = rs.std_profit / abs(rs.avg_profit) if rs.avg_profit != 0 else 0.0
            else:
                top_10_profit = 0.0
                bottom_10_profit = 0.0
                cv = 0.0
            
            writer.writerow([
                rs.round_num,
                f"{rs.market_price:.6f}",
                f"{rs.gini_coefficient:.6f}",
                f"{rs.max_profit - rs.min_profit:.6f}",
                f"{rs.std_profit:.6f}",
                f"{cv:.6f}",
                f"{top_10_profit:.6f}",
                f"{bottom_10_profit:.6f}",
            ])
    artifacts["inequality"] = str(ineq_path)
    
    return artifacts


def export_profitability_summary(
    session: ProfitabilitySession,
    *,
    out_dir: str = "outputs",
    run_name: str = "profitability_run",
) -> str:
    """
    Export human-readable summary of profitability metrics.
    
    Returns:
        Path to summary file
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)
    
    summary_path = out_path / f"{run_name}_summary.txt"
    
    final_round = session.round_snapshots[-1] if session.round_snapshots else None
    
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PROFITABILITY ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        summary = session.get_summary()
        f.write(f"Simulation ID: {summary.get('simulation_id', 'N/A')}\n")
        f.write(f"Total Rounds: {summary.get('total_rounds', 0)}\n")
        f.write(f"Number of Agents: {summary.get('num_agents', 0)}\n")
        f.write(f"Final Market Price: {summary.get('final_market_price', 0):.6f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("FINAL ROUND PROFITABILITY METRICS\n")
        f.write("-" * 80 + "\n")
        
        if final_round:
            f.write(f"Average Profit: ${final_round.avg_profit:.2f}\n")
            f.write(f"Median Profit: ${final_round.median_profit:.2f}\n")
            f.write(f"Std Dev of Profit: ${final_round.std_profit:.2f}\n")
            f.write(f"Max Profit: ${final_round.max_profit:.2f}\n")
            f.write(f"Min Profit: ${final_round.min_profit:.2f}\n")
            f.write(f"Profit Range: ${final_round.max_profit - final_round.min_profit:.2f}\n")
            f.write(f"Gini Coefficient (Inequality): {final_round.gini_coefficient:.4f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("BELIEF ACCURACY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Belief Error: {final_round.avg_belief_error:.6f}\n")
            f.write(f"Belief-Accuracy Premium (Correlation): {final_round.belief_accuracy_premium:.4f}\n")
            f.write("  (Positive = accurate beliefs → higher profits)\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("TOP 5 AGENTS BY PROFIT\n")
            f.write("-" * 80 + "\n")
            sorted_agents = sorted(
                final_round.agent_snapshots,
                key=lambda a: a.total_pnl,
                reverse=True
            )
            for i, agent in enumerate(sorted_agents[:5], 1):
                f.write(
                    f"{i}. Agent {agent.agent_id} (rho={agent.rho:.2f}): "
                    f"${agent.total_pnl:.2f} "
                    f"({agent.shares:.1f} shares @ ${agent.market_price:.6f})\n"
                )
            
            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("BOTTOM 5 AGENTS BY PROFIT\n")
            f.write("-" * 80 + "\n")
            for i, agent in enumerate(reversed(sorted_agents[-5:]), 1):
                f.write(
                    f"{i}. Agent {agent.agent_id} (rho={agent.rho:.2f}): "
                    f"${agent.total_pnl:.2f} "
                    f"({agent.shares:.1f} shares @ ${agent.market_price:.6f})\n"
                )
    
    return str(summary_path)
