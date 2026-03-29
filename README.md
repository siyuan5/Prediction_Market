# Prediction Market — Deliverable 2 (Team A: LMSR)

This repository contains code for simulating prediction markets. Team A uses the **LMSR (Logarithmic Market Scoring Rule)** mechanism.

## Deliverable 2 Overview

There are two phases for Deliverable 2. This section documents both phases **for Team A (LMSR)** — the code here does **not** describe Team B (CDA/auction).

---

## Phase 1: Static Beliefs

- **Agents**: Each agent starts with a fixed belief about the ground truth probability P* (e.g., P* = 0.70).
- **Behavior**: Agents do _not_ update their beliefs during trading.
- **Market Mechanism**: LMSR (automated market maker).
- **Goal**: 
  - Does the market price converge toward the **mean initial belief**?
  - How does changing **rho** (risk aversion) affect **final position sizes** (number of shares/cash held by agents)?

#### Run & Export (Phase 1)

From the repo root, run:

```bash
python src/team_a_main.py
```

Outputs will appear in `outputs/`, including:
- Time series files (`<run>_timeseries.csv`)
- Per-agent results (`<run>_agents.csv`)
- Rho summaries (`<run>_rho_summary.csv`)
- Full JSON state (`<run>.json`)

*See "Output files" below for details on these exports.*

---

## Phase 2: Updating Beliefs (Learning from Noisy Signals)

**Phase 2 investigates how agents and the market learn the ground truth P\* over time.**

- **Ground Truth**: P\* remains fixed throughout the simulation.
- **New Information Each Round**:  
  At each round *t*, every agent receives a noisy signal Sₜ (e.g., a "poll", often a binomial sample related to P\*).
- **Belief Updates**:
  - Agents update their internal probability estimate (pi) using Bayesian updating (see `run_phase2()` and `phase2_utils.py`), giving weighted consideration to prior and observed signals.
  - Update method and weights are set via parameters (e.g., `prior_strength`, `obs_strength`, `signal_spec`).
- **Trading**: 
  - After updating beliefs, agents trade in the LMSR market.
- **Goal**:
  - Observe how quickly the market price tracks the “discovery” of P\* as agents receive and respond to new information.

### How to Run Team A Phase 2

From the repo root:

```bash
python run_team_a_phase2.py
```

This will:
- Simulate a market where agents **continually update their beliefs from signals** and trade.
- Output the same set of files as Phase 1, labeled as `team_a_phase2_baseline.*` in `outputs/`.
- Print key summary statistics (final price, error, summary of saved file paths).

### What Phase 2 Outputs Mean

Output files have similar structure/format as in Phase 1. Some especially relevant columns to observe:
- `<run>_timeseries.csv`:
  - `price`
  - `abs_error_vs_ground_truth`: Market price tracking P\* over time
- `<run>_agents.csv`:
  - Final agent beliefs, shares, and cash at the end of learning
- `<run>.json`:
  - Full arrays of agent beliefs and prices throughout, for custom analysis

---

## Setup

Requires Python 3.9+.

```bash
pip install numpy
```

---

## Output files (how to interpret)

For each run name `<run>`:

### 1) `<run>_timeseries.csv`
Per-round time series:
- `price`: LMSR market price each round
- `abs_error_vs_ground_truth`: `|price - P*|` per round (convergence tracking)
- `trade_volume`: total absolute shares traded in that round

Use this for:
- “Does price converge toward mean belief?” (Phase 1)
- “Does error shrink over time as agents learn?” (Phase 2, observe how quickly price tracks P*)

### 2) `<run>_agents.csv`
One row per agent:
- `rho`: risk aversion
- `final_shares`, `final_cash`: final portfolio after trading

Use this for:
- Comparing position sizes across rho values
- Observing how learning affects final agent holdings (Phase 2)

### 3) `<run>_rho_summary.csv`
Aggregated by rho:
- `avg_shares`, `avg_cash`

Use this for quick analysis of risk aversion effects.

### 4) `<run>.json`
Full raw results, including all arrays underlying the CSVs.

---

## Web UI (React) + API (Optional)

The `frontend/` app talks to `api/main.py`. You can run both from the repo root (two terminals):

**Backend** — install dependencies and launch API:

```bash
pip install -r requirements-api.txt
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend** — install and start dev server (proxies `/api` to port 8000):

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite (typically `http://127.0.0.1:5173`). For production, build with `npm run build` and serve `frontend/dist/`.

---

## Notes
- **Ground truth** `P*` is controlled by the `ground_truth` parameter in `run_phase1` or `run_phase2` (see `src/team_a_phase1_simulation.py` and `team_a_phase2_simulation.py`).
- **Liquidity** is controlled by `b`: higher `b` = less price movement per trade.
- **Signal/learning settings** for Phase 2 (e.g., number of signals, update strengths) are set in `run_team_a_phase2.py`.