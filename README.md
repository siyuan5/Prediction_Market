# Prediction Market — Deliverable 2 (Phase 1)

Deliverable 2 asks you to run **Phase 1 (Static Beliefs)** with a fixed ground truth (e.g., P* = 0.70), then:
- check whether market price converges toward the **mean initial belief**
- analyze how changing **rho (risk aversion)** affects **final position sizes** (shares/cash)

This repo contains Team A (LMSR) and Team B (CDA). The instructions below cover **Team A Phase 1 exports**.

## Setup

Requires Python 3.9+.

```bash
pip install numpy
```

## Run Team A Phase 1 (and export CSV/JSON)

From the repo root:

```bash
python src/team_a_main.py
```

This will run several Phase 1 test configurations and write artifacts into:

```
outputs/
  test1_baseline.json
  test1_baseline_timeseries.csv
  test1_baseline_agents.csv
  test1_baseline_rho_summary.csv
  ...
```

## Output files (how to interpret)

For each run name `<run>`:

### 1) `<run>_timeseries.csv`
Per-round time series:
- `price`: LMSR market price each round
- `abs_error_vs_ground_truth`: `|price - P*|` per round (convergence tracking)
- `trade_volume`: total absolute shares traded in that round

Use this for:
- “Does price converge toward mean belief?”
- “Does error shrink over time?”

### 2) `<run>_agents.csv`
One row per agent:
- `rho`: risk aversion
- `final_shares`, `final_cash`: final portfolio after Phase 1

Use this for:
- comparing position sizes across rho values (e.g., average absolute shares by rho)

### 3) `<run>_rho_summary.csv`
Aggregated by rho:
- `avg_shares`, `avg_cash`

Use this for the deliverable’s rho analysis quickly.

### 4) `<run>.json`
Full raw results (includes the arrays used to build the CSVs).

## Notes
- Ground truth `P*` is controlled by the `ground_truth` parameter in `run_phase1` (see `src/team_a_phase1_simulation.py`).
- Liquidity is controlled by `b` (higher b generally means prices move less per unit trade).
