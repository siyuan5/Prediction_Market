# Copilot instructions for Prediction_Market

## Project overview
- This is a small Python simulation/prototype for a prediction market using LMSR and CRRA agents.
- Core market math lives in [src/market_logic.py](src/market_logic.py) in `LMSRMarketMaker`.
- Agent behavior is modeled in [src/crra_agent.py](src/crra_agent.py) in `CRRAAgent`, which computes optimal trades and updates portfolios.
- There are two minimal entry scripts: [main.py](main.py) and [src/main.py](src/main.py), both currently just print placeholders.

## Architecture & data flow
- `LMSRMarketMaker` maintains an internal `inventory` vector and exposes:
  - `get_price()` to compute the current outcome-1 price from inventory.
  - `calculate_trade_cost(delta_q1)` to update inventory and return the cost of a trade.
- `CRRAAgent.get_optimal_trade(market_price)` computes the desired share change given belief, cash, shares, and risk aversion. It uses safety checks (cash- and margin-like constraints) before returning the trade size.
- There is no package layout; modules are imported by filename (e.g., `from market_logic import LMSRMarketMaker` in [tests/test_market.py](tests/test_market.py)). Keep imports consistent with this flat-module style unless you convert the project into a package.

## Dependencies & environment
- Dependencies are managed in [pyproject.toml](pyproject.toml); only `numpy` is required.
- The `uv.lock` file suggests the project may be managed with uv, but no workflow is documented yet.

## Developer workflows (observed)
- There is no pytest-style test suite; [tests/test_market.py](tests/test_market.py) is a runnable script that prints success/failure based on price movement.
- To exercise the market logic, run the script file directly with Python (no CLI or task runner is defined).

## Conventions & patterns
- Numerical computation uses `numpy` arrays and math (see [src/market_logic.py](src/market_logic.py)).
- Safety checks and boundary conditions are handled explicitly inside `CRRAAgent.get_optimal_trade()` (see [src/crra_agent.py](src/crra_agent.py)). Preserve these checks if modifying trading logic.
- LMSR inventory is stored as a length-2 vector `[q1, q0]` (implicit from usage in [src/market_logic.py](src/market_logic.py)). Keep this ordering consistent.
