import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_phase1_results(
    results: Dict[str, Any],
    *,
    out_dir: str = "outputs",
    run_name: str = "phase1_run",
) -> Dict[str, str]:
    """Export a Phase 1 run (dict returned by run_phase1) into JSON + CSVs.

    Writes:
      - <run_name>.json (full results)
      - <run_name>_timeseries.csv (round, price, error, volume)
      - <run_name>_agents.csv (agent_id, rho, final_shares, final_cash)
      - <run_name>_rho_summary.csv (rho, avg_shares, avg_cash)

    Returns a map of artifact name -> path string.
    """
    out_path = Path(out_dir)
    _ensure_dir(out_path)

    # ---------- JSON (full results) ----------
    json_path = out_path / f"{run_name}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ---------- Time series ----------
    ts_path = out_path / f"{run_name}_timeseries.csv"
    price_series: List[float] = list(results.get("price_series", []))
    error_series: List[float] = list(results.get("error_series", []))
    trade_volume: List[float] = list(results.get("trade_volume", []))

    n = max(len(price_series), len(error_series), len(trade_volume))
    with ts_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "price", "abs_error_vs_ground_truth", "trade_volume"])
        for t in range(n):
            price = price_series[t] if t < len(price_series) else ""
            err = error_series[t] if t < len(error_series) else ""
            vol = trade_volume[t] if t < len(trade_volume) else ""
            w.writerow([t, price, err, vol])

    # ---------- Agent finals ----------
    agents_path = out_path / f"{run_name}_agents.csv"
    rhos: List[float] = list(results.get("final_rhos", []))
    shares: List[float] = list(results.get("final_positions", []))
    cash: List[float] = list(results.get("final_cash", []))

    m = max(len(rhos), len(shares), len(cash))
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

    #  rho summary 
    rho_sum_path = out_path / f"{run_name}_rho_summary.csv"
    rho_summary: Dict[str, Dict[str, float]] = results.get("rho_summary", {})  # type: ignore
    # keys might be floats; normalize for sorting
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
