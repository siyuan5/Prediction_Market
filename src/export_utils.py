import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import numpy as np


def _ensure_dir(path: Path) -> None:
    """Create directory (including parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def _build_rho_summary_from_arrays(results: Dict[str, Any]) -> Dict[float, Dict[str, float]]:
    """
    Compute per-rho statistics (average shares and cash) from flat arrays.
    Returns a mapping: rho value -> {"avg_shares": ..., "avg_cash": ...}

    Meant for runs that do not compute rho summary upfront.
    """
    rhos: List[float] = list(results.get("final_rhos", []))
    shares: List[float] = list(results.get("final_positions", []))
    cash: List[float] = list(results.get("final_cash", []))

    # Group agent results by unique rho value
    rho_groups: Dict[float, Dict[str, List[float]]] = {}
    n = min(len(rhos), len(shares), len(cash))
    for i in range(n):
        rho = float(rhos[i])
        if rho not in rho_groups:
            rho_groups[rho] = {"shares": [], "cash": []}
        rho_groups[rho]["shares"].append(float(shares[i]))
        rho_groups[rho]["cash"].append(float(cash[i]))

    # For each rho, compute group mean for shares and cash
    rho_summary: Dict[float, Dict[str, float]] = {}
    for rho, data in rho_groups.items():
        rho_summary[rho] = {
            "avg_shares": float(np.mean(data["shares"])) if data["shares"] else 0.0,
            "avg_cash": float(np.mean(data["cash"])) if data["cash"] else 0.0,
        }
    return rho_summary


def export_phase1_results(
    results: Dict[str, Any],
    *,
    out_dir: str = "outputs",
    run_name: str = "phase1_run",
) -> Dict[str, str]:
    """
    Export full results of a Phase 1 simulation run (see run_phase1) as JSON 
    and relevant CSV files. Returns mapping: artifact label -> file path.

    Output files:
      - <run_name>.json:      Complete results (raw dict)
      - <run_name>_timeseries.csv:  Market price/error/volume per round
      - <run_name>_agents.csv:      Agent final (rho, shares, cash)
      - <run_name>_rho_summary.csv: Per-rho aggregated agent stats
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    # ---- Serialize full run to JSON ----
    json_path = out_path / f"{run_name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- Time series: one row per round ----
    ts_path = out_path / f"{run_name}_timeseries.csv"
    price_series: List[float] = list(results.get("price_series", []))
    error_series: List[float] = list(results.get("error_series", []))
    trade_volume: List[float] = list(results.get("trade_volume", []))

    n = max(len(price_series), len(error_series), len(trade_volume))  # ensures correct # of rounds written
    with ts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "price", "abs_error_vs_ground_truth", "trade_volume"])
        for t in range(n):
            price = price_series[t] if t < len(price_series) else ""
            err = error_series[t] if t < len(error_series) else ""
            vol = trade_volume[t] if t < len(trade_volume) else ""
            w.writerow([t, price, err, vol])

    # ---- Agent outcomes: one row per agent ----
    agents_path = out_path / f"{run_name}_agents.csv"
    rhos: List[float] = list(results.get("final_rhos", []))
    shares: List[float] = list(results.get("final_positions", []))
    cash: List[float] = list(results.get("final_cash", []))

    m = max(len(rhos), len(shares), len(cash))  # ensure rows for all tracked arrays
    with agents_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_id", "rho", "final_shares", "final_cash"])
        for i in range(m):
            w.writerow([
                i,
                rhos[i] if i < len(rhos) else "",
                shares[i] if i < len(shares) else "",
                cash[i] if i < len(cash) else "",
            ])

    # ---- Per-rho summary (average per group) ----
    rho_sum_path = out_path / f"{run_name}_rho_summary.csv"
    # If precomputed, use it; else, derive from agent arrays
    rho_summary: Dict[Any, Dict[str, float]] = results.get("rho_summary", {})  # type: ignore
    if not rho_summary:
        rho_summary = _build_rho_summary_from_arrays(results)

    # Normalize keys (rho) for sorting in CSV output
    items = sorted(rho_summary.items(), key=lambda kv: float(kv[0]))
    with rho_sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rho", "avg_shares", "avg_cash"])
        for rho, summary in items:
            w.writerow([rho, summary.get("avg_shares", ""), summary.get("avg_cash", "")])

    return {
        "json": str(json_path),
        "timeseries_csv": str(ts_path),
        "agents_csv": str(agents_path),
        "rho_summary_csv": str(rho_sum_path),
    }


def export_phase2_results(
    results: Dict[str, Any],
    *,
    out_dir: str = "outputs",
    run_name: str = "phase2_run",
) -> Dict[str, str]:
    """
    Export Phase 2 simulation results as JSON and several detailed CSV artifacts:
      - full results
      - per time step (including mean belief, signal, and inventory position metrics)
      - agent outcomes
      - rho summary
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    # Write main JSON (all outputs)
    json_path = out_path / f"{run_name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---- Export per-round series ----
    ts_path = out_path / f"{run_name}_timeseries.csv"
    price_series: List[float] = list(results.get("price_series", []))
    error_series: List[float] = list(results.get("error_series", []))
    trade_volume: List[float] = list(results.get("trade_volume", []))
    signal_series: List[float] = list(results.get("signal_series", []))
    mean_belief_series: List[float] = list(results.get("mean_belief_series", []))
    inventory_series: List[List[float]] = list(results.get("inventory_series", []))

    # Number of rounds taken as max of all available metrics (robust to missing data)
    n = max(
        len(price_series),
        len(error_series),
        len(trade_volume),
        len(signal_series),
        len(mean_belief_series),
        len(inventory_series),
    )

    with ts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "round",
                "price",
                "signal",
                "mean_belief",
                "abs_error_vs_ground_truth",
                "trade_volume",
                "inventory_q1",
                "inventory_q0",
            ]
        )
        for t in range(n):
            price = price_series[t] if t < len(price_series) else ""
            signal = signal_series[t] if t < len(signal_series) else ""
            mean_belief = mean_belief_series[t] if t < len(mean_belief_series) else ""
            err = error_series[t] if t < len(error_series) else ""
            vol = trade_volume[t] if t < len(trade_volume) else ""

            # LMSR market may have inventory for both Q=1 and Q=0 outcomes
            q1 = ""
            q0 = ""
            if t < len(inventory_series):
                inv = inventory_series[t]
                if isinstance(inv, list) and len(inv) >= 2:
                    q1 = inv[0]  # shares in Q=1 state
                    q0 = inv[1]  # shares in Q=0 state

            w.writerow([t, price, signal, mean_belief, err, vol, q1, q0])

    # ---- Export agent outcomes (final) ----
    agents_path = out_path / f"{run_name}_agents.csv"
    rhos: List[float] = list(results.get("final_rhos", []))
    shares: List[float] = list(results.get("final_positions", []))
    cash: List[float] = list(results.get("final_cash", []))
    beliefs: List[float] = list(results.get("final_beliefs", []))

    m = max(len(rhos), len(shares), len(cash), len(beliefs))
    with agents_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_id", "rho", "final_belief", "final_shares", "final_cash"])
        for i in range(m):
            w.writerow(
                [
                    i,
                    rhos[i] if i < len(rhos) else "",
                    beliefs[i] if i < len(beliefs) else "",
                    shares[i] if i < len(shares) else "",
                    cash[i] if i < len(cash) else "",
                ]
            )

    # ---- Write rho-aggregated summary from agent outcomes ----
    rho_sum_path = out_path / f"{run_name}_rho_summary.csv"
    rho_summary: Dict[float, Dict[str, float]] = _build_rho_summary_from_arrays(results)
    items = sorted(rho_summary.items(), key=lambda kv: float(kv[0]))
    with rho_sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rho", "avg_shares", "avg_cash"])
        for rho, summary in items:
            w.writerow([rho, summary.get("avg_shares", ""), summary.get("avg_cash", "")])

    return {
        "json": str(json_path),
        "timeseries_csv": str(ts_path),
        "agents_csv": str(agents_path),
        "rho_summary_csv": str(rho_sum_path),
    }
