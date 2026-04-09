# Final Phase: Autonomous Agent Architecture

Target: Working autonomous trading system for final presentation (~1 month out).

---

## Design Decisions (Locked)

| Question | Decision |
|---|---|
| Async model | Simulated async — threads in one Python process, not multi-process |
| Number of markets | Single market at a time, but schema supports `market_id` for future multi-market |
| Round-based mode | Kept alongside autonomous mode, both paths coexist |
| Persistence | In-memory SQLite (`sqlite3.connect(":memory:", check_same_thread=False)`), fresh per run |
| HTTP client for agents | `requests` library (sync), not `aiohttp` |
| Concurrency | **One `threading.Lock` per market** — stored in `MarketDB` as `self._locks[market_id]`. Every write goes through `with self._locks[market_id]:` |
| CRRA math location | Extract shared helper to `src/crra_math.py` with `compute_optimal_trade(belief, price, cash, shares, rho) -> float`. Both `crra_agent.py` and `autonomous_agent.py` import from it |

---

## Locked API Contracts

Every endpoint returns JSON. Errors return `{"error": "message"}` with appropriate HTTP status.

### POST /api/market/create

**Request:**
```json
{
  "mechanism": "lmsr",
  "ground_truth": 0.70,
  "n_agents": 50,
  "initial_cash": 100.0,
  "b": 100.0,
  "belief_spec": {"mode": "gaussian", "sigma": 0.10},
  "rho_distribution": {"min": 0.5, "max": 2.0},
  "personality_defaults": { /* see personality schema below */ }
}
```

**Response (201):**
```json
{
  "market_id": "m_abc123",
  "mechanism": "lmsr",
  "n_agents": 50,
  "initial_price": 0.5,
  "ground_truth": 0.70
}
```

**Errors:** 400 if mechanism invalid, n_agents < 2, or ground_truth out of [0.01, 0.99].

---

### GET /api/market/{market_id}/price

**Response (200):**
```json
{
  "market_id": "m_abc123",
  "price": 0.6847,
  "best_bid": 0.68,
  "best_ask": 0.69,
  "last_trade_price": 0.685,
  "last_trade_at": "2026-04-08T14:32:10.123Z",
  "timestamp": "2026-04-08T14:32:11.456Z"
}
```

For LMSR, `best_bid` and `best_ask` are `null`.

**Errors:** 404 if market_id not found.

---

### GET /api/market/{market_id}/book

Only valid for CDA markets.

**Response (200):**
```json
{
  "bids": [{"price": 0.68, "quantity": 12.5}, {"price": 0.67, "quantity": 8.0}],
  "asks": [{"price": 0.69, "quantity": 15.0}, {"price": 0.70, "quantity": 5.2}]
}
```

**Errors:** 404 if market_id not found. 400 if market is LMSR.

---

### POST /api/market/{market_id}/trade

**Request (LMSR):**
```json
{
  "agent_id": 7,
  "quantity": 3.5
}
```
Positive = buy Yes, negative = sell Yes.

**Request (CDA):**
```json
{
  "agent_id": 7,
  "side": "buy",
  "quantity": 3.5,
  "limit_price": 0.68,
  "order_type": "limit"
}
```
`order_type` is `"limit"` or `"market"`. For market orders, `limit_price` is ignored.

**Response (200):**
```json
{
  "trade_id": "t_xyz789",
  "executed_quantity": 3.5,
  "executed_price": 0.6847,
  "cost": 2.396,
  "new_price": 0.6891,
  "agent_cash_after": 97.604,
  "agent_shares_after": 3.5
}
```

**Errors:**
- 400 if quantity is 0, agent has insufficient cash, agent has insufficient shares to sell
- 404 if market or agent not found
- 409 if market is stopped

---

### GET /api/market/{market_id}/agent/{agent_id}

**Response (200):**
```json
{
  "agent_id": 7,
  "cash": 97.604,
  "shares": 3.5,
  "belief": 0.72,
  "rho": 1.0,
  "pnl": 2.012,
  "personality": { /* see schema below */ }
}
```

**Errors:** 404 if market or agent not found.

---

### POST /api/market/{market_id}/agent/{agent_id}/belief

**Request:**
```json
{"new_belief": 0.85}
```
OR
```json
{"delta": -0.10}
```

Exactly one of `new_belief` or `delta` must be provided.

**Response (200):**
```json
{
  "agent_id": 7,
  "old_belief": 0.72,
  "new_belief": 0.85,
  "at_timestamp": "2026-04-08T14:32:15.789Z"
}
```

**Errors:** 400 if both or neither field given, or if belief out of [0.01, 0.99]. 404 if not found.

---

### GET /api/market/{market_id}/agents

**Response (200):** Array of agent objects (same shape as single-agent endpoint).

**Query params:** `?limit=50&offset=0` for pagination.

---

### GET /api/market/{market_id}/trades

**Response (200):**
```json
{
  "trades": [
    {"trade_id": "t_001", "agent_id": 7, "quantity": 3.5, "price": 0.6847, "at": "..."},
    ...
  ],
  "total": 142
}
```

**Query params:** `?since=<trade_id>` returns only trades after the given id. `?limit=100` caps response size (default 100).

---

### POST /api/market/{market_id}/start

Starts the autonomous agent runner threads.

**Response (200):**
```json
{"status": "started", "n_agents_running": 50}
```

**Errors:** 409 if market already running.

---

### POST /api/market/{market_id}/stop

**Response (200):**
```json
{"status": "stopped", "total_trades": 342, "duration_sec": 30.1}
```

**Errors:** 409 if market not running.

---

## Personality JSON Schema

```json
{
  "check_interval_mean": 2.0,
  "check_interval_jitter": 1.0,
  "edge_threshold": 0.03,
  "participation_rate": 0.80,
  "trade_size_noise": 0.20,
  "signal_sensitivity": 0.50,
  "stubbornness": 0.30
}
```

**Field definitions:**
- `check_interval_mean` (seconds, >0): average time between market polls
- `check_interval_jitter` (seconds, >=0): uniform random noise added to interval, so actual = mean ± jitter
- `edge_threshold` ([0, 1]): minimum absolute gap between belief and price before considering a trade
- `participation_rate` ([0, 1]): probability of actually placing a trade when threshold is met
- `trade_size_noise` ([0, 1]): multiplier range on optimal trade size, so actual = optimal * uniform(1 - noise, 1 + noise)
- `signal_sensitivity` ([0, 1]): weight applied to new signals during belief updates (1.0 = full update, 0.0 = ignore)
- `stubbornness` ([0, 1]): decay factor dampening belief changes (1.0 = no change, 0.0 = no dampening)

**Default population distribution** (when `personality_defaults` is not overridden):
- `check_interval_mean` drawn from `Uniform(1.0, 4.0)`
- `edge_threshold` drawn from `Uniform(0.01, 0.10)`
- `participation_rate` drawn from `Uniform(0.50, 1.0)`
- Everything else fixed at the defaults above

---

## Error Handling Checklist

Every task must handle these:

| Scenario | Expected behavior |
|---|---|
| Agent tries to buy more than cash allows | Clip quantity to max affordable, log, execute clipped trade |
| Agent tries to sell more shares than held | Clip to available shares, log, execute clipped trade |
| Trade arrives for a stopped market | Return 409, agent client waits 1s and retries |
| Concurrent trades on same market | Serialized by market lock, later trade sees updated price |
| Agent thread crashes | Runner logs exception, restarts that agent after 2s |
| DB write fails mid-trade | Rollback transaction, return 500, no partial state |
| Belief shift while agent is mid-trade-computation | Agent re-reads belief at start of next cycle, no mid-cycle abort |
| Market stopped while agents running | Runner sets stop flag, agents exit cleanly within next poll cycle |

---

## Task 1 — Persistent Market State Layer

**File:** `src/market_db.py`

**Build:**
- `MarketDB` class with `sqlite3` in-memory connection, `check_same_thread=False`
- Per-market lock dict: `self._locks: dict[str, threading.Lock] = {}`
- Schema creation on init (all tables listed in overview)
- Methods (all thread-safe):
  - `create_market(mechanism, ground_truth, b, tick_size) -> market_id`
  - `create_agent(market_id, cash, belief, rho, personality) -> agent_id`
  - `get_price(market_id) -> dict`
  - `update_price(market_id, new_price, inventory_update)`
  - `get_agent(market_id, agent_id) -> dict`
  - `update_agent_portfolio(market_id, agent_id, cash_delta, shares_delta)`
  - `set_agent_belief(market_id, agent_id, new_belief) -> old_belief`
  - `insert_order(market_id, agent_id, side, quantity, limit_price, order_type) -> order_id`
  - `get_order_book(market_id) -> dict` (bids/asks)
  - `record_trade(market_id, agent_id, quantity, price) -> trade_id`
  - `get_trades(market_id, since_trade_id=None, limit=100) -> list`
  - `set_market_status(market_id, status)` where status in {"created", "running", "stopped"}

**Done when:**
- `pytest tests/test_market_db.py` passes with cases: create market, insert agents, submit trades concurrently from 10 threads without data corruption, read final state matches expected cash/shares totals

---

## Task 2 — Thread-Safe Market Service

**File:** `src/market_service.py`

**Build:**
- `MarketService` class with dependency on `MarketDB`
- `execute_lmsr_trade(market_id, agent_id, quantity) -> dict` — atomic under market lock: read price, compute cost, validate agent cash/shares, update agent portfolio, update market inventory, update price, record trade, return result dict
- `execute_cda_order(market_id, agent_id, side, quantity, limit_price, order_type) -> dict` — atomic: insert order, run matching, execute resulting trades, update both counterparty portfolios, update price (mid or last-trade), return result dict
- `get_price_snapshot(market_id) -> dict`
- `get_order_book(market_id) -> dict`
- Imports CRRA cost logic from existing `team_a_market_logic.py` and order-matching from `team_b_market_logic.py` — does not re-implement

**Done when:**
- `pytest tests/test_market_service.py` passes with cases: 10 concurrent `execute_lmsr_trade` calls produce consistent final state, CDA order matching works with crossing orders, clipping works when agent has insufficient cash

---

## Task 3 — Agent-Facing API Endpoints

**File:** `api/main.py` (modify, don't break existing endpoints)

**Build:**
- Import `MarketService` as a module-level singleton
- Add all endpoints listed in "Locked API Contracts" section above
- Each endpoint validates input, calls service, returns dict (FastAPI handles JSON)
- Use Pydantic models for request validation — create `MarketCreateRequest`, `TradeRequest`, `BeliefUpdateRequest` classes
- Response models optional but recommended

**Done when:**
- Manual curl flow works: create market → query price → submit trade → verify new price reflects trade → query agent state → shift belief → verify new belief
- All existing session endpoints still respond correctly (regression check)
- `pytest tests/test_api_endpoints.py` covers every new endpoint including error cases

---

## Task 4 — Autonomous Agent Client

**Files:** `src/autonomous_agent.py`, `src/crra_math.py` (new shared helper)

**Build:**

First, extract shared CRRA logic:
- `src/crra_math.py` with `compute_optimal_trade(belief, price, cash, shares, rho) -> float`
- Update `src/crra_agent.py` to import and call this
- All existing tests must still pass

Then build the agent:
- `AutonomousAgent` class with `__init__(agent_id, market_id, api_base_url, personality)`
- `self._stop_flag = threading.Event()`
- `run()` method — loop until stop flag:
  1. Sleep `uniform(mean - jitter, mean + jitter)` seconds
  2. GET `/market/{id}/price` → extract current price
  3. GET `/market/{id}/agent/{id}` → extract cash, shares, belief
  4. Compute `x_star = compute_optimal_trade(belief, price, cash, shares, rho)`
  5. Apply trade_size_noise: `x_star *= uniform(1 - noise, 1 + noise)`
  6. If `abs(belief - price) < edge_threshold`, skip
  7. If `random() > participation_rate`, skip
  8. POST `/market/{id}/trade` with quantity
  9. Log result (or error, with retry on 409)
- `stop()` sets the event flag
- Use `requests.Session()` for connection reuse

**Done when:**
- `pytest tests/test_autonomous_agent.py` passes with mocked HTTP responses
- Manual test: start the API server, instantiate one `AutonomousAgent`, run for 10 seconds, observe at least 3 trades in the trade feed
- `crra_agent.py` tests all still pass after math extraction

---

## Task 5 — Agent Personality System

**Files:** `src/personality.py`, modify `src/autonomous_agent.py`

**Build:**
- `Personality` dataclass with all 7 fields from schema, serializable via `asdict()`
- `sample_personality(distribution_config, rng) -> Personality` — samples each field from its configured distribution
- `DEFAULT_POPULATION_DIST` constant matching the spec above
- Wire `AutonomousAgent.run()` to use all personality fields (steps 1, 5, 6, 7 above)
- At market creation time in Task 3, each agent gets a sampled personality stored in DB

**Done when:**
- Two agents with identical beliefs/cash/rho but different personalities demonstrably trade differently in a 30-second run (different trade counts, different average trade sizes)
- Unit tests cover all 7 personality fields affecting behavior
- Serialization round-trip works: `Personality -> dict -> JSON -> dict -> Personality`

---

## Task 6 — Multi-Threaded Agent Orchestrator

**File:** `src/agent_runner.py`

**Build:**
- `AgentRunner` class owned by the API layer (one per market)
- `__init__(market_id, api_base_url, market_db)`
- `start_all()` — reads all agents from DB, instantiates `AutonomousAgent` for each, spawns daemon thread for each, stores thread handles
- `stop_all()` — sets stop flags, joins threads with 5-second timeout, force-kills any stuck threads, updates market status in DB
- `is_running() -> bool`
- `agent_count_active() -> int`
- Thread-restart logic: if any agent thread dies unexpectedly, log and restart it (within reason — cap at 3 restarts per agent)
- Wire to `/api/market/{id}/start` and `/stop` endpoints

**Done when:**
- POST start → 50 agents trade for 30 seconds → POST stop → all threads exit cleanly within timeout
- `ps` / thread dump shows no zombie threads after stop
- Killing one agent mid-run triggers a clean restart, final state consistent

---

## Task 7 — Frontend Autonomous Mode Integration

**Files:** `frontend/src/App.tsx`, new components under `frontend/src/components/`

**Build:**
- Top-level toggle: `Round Mode | Autonomous Mode` — hides/shows respective panels
- Autonomous mode panel components:
  - `MarketSetup` — form for mechanism, n_agents, ground_truth, personality distribution, "Create Market" button
  - `MarketControls` — Start/Stop buttons, status indicator, elapsed time, trade count
  - `LivePriceChart` — polls `/market/{id}/price` every 500ms, appends to a rolling 200-point buffer
  - `TradeFeed` — polls `/market/{id}/trades?since=<last_id>` every 1s, shows last 20 in a scrollable list
  - `AgentTable` — polls `/market/{id}/agents` every 2s, shows belief, cash, shares, pnl, sortable columns
  - `BeliefShiftPanel` — select agent (or "all"), input new belief, POST to the shift endpoint, show confirmation
- Round mode remains fully functional, no regressions
- Use `useEffect` + `setInterval` for polling, clean up on unmount

**Done when:**
- PM can walk through full demo: create market → start → watch live chart and trade feed → shift beliefs for a subset of agents → observe price reaction → stop → see final agent table
- Round mode still works identically to before
- No console errors in either mode

---

## Task 8 — Autonomous Mode Evaluation

**Files:** `run_autonomous_benchmark.py`, `outputs/autonomous/*`

**Build:**

Three experiments, each exporting CSV + JSON + chart:

**Experiment 1 — Autonomous vs Round-Based Convergence**
- Same seed, same n_agents (50), same ground_truth (0.70), same belief_spec (gaussian, σ=0.1)
- Run autonomous for 60 seconds, run round-based for 100 rounds
- Compare: final price, final error, trajectory shape
- Success criterion: autonomous converges within 0.05 of round-based final price

**Experiment 2 — Personality Diversity Effect**
- Three configs: all agents identical, moderately varied, highly varied
- Run autonomous for 60 seconds each, 5 seeds per config
- Compare: final error, trade count, price stability (rolling std)
- Deliverable: 3-column table with mean ± std per metric

**Experiment 3 — Belief Shock Recovery**
- Run autonomous for 30 seconds (baseline period)
- Shift beliefs of 50% of agents by +0.20
- Run another 30 seconds (recovery period)
- Measure: time for price to move 50% of the way to new mean belief, final error vs new mean belief
- Deliverable: overlay chart showing pre/post-shock price trajectory

**Done when:**
- All three experiments run end-to-end with one command per experiment
- Output artifacts exist in `outputs/autonomous/` (CSVs, JSON, matplotlib PNGs)
- Summary table ready to drop into the final deliverable and slides

---

## Task 9 — Unit Test Coverage for New Modules (Distributed)

Each developer writes unit tests for the module they own as part of their task. This is not a separate assignment — it's a requirement baked into Tasks 1 through 7.

**Per-task test files:**
- Task 1 owner → `tests/test_market_db.py`
- Task 2 owner → `tests/test_market_service.py`
- Task 3 owner → `tests/test_api_endpoints.py`
- Task 4 owner → `tests/test_autonomous_agent.py` + updates to `tests/test_crra_agent.py` after math extraction
- Task 5 owner → `tests/test_personality.py`
- Task 6 owner → `tests/test_agent_runner.py`
- Task 7 owner → minimal smoke tests for new React components (optional but encouraged)

**Done when:** every new module has unit tests covering normal cases and the error scenarios listed in the Error Handling Checklist.

---

## Task 10 — End-to-End Integration and Regression Validation

**Files:** `tests/test_integration.py`

**Owner:** Coordinator (separate from Task 9 which is distributed across devs).

**Build:**
- Full-stack integration tests that spin up the API server, create a market, run autonomous agents, verify state consistency end to end
- Belief shock integration test: start market, run 10 seconds, inject shift, run another 10 seconds, assert price moved toward new belief mean
- Concurrency stress test: 50 agents running for 30 seconds, assert no data corruption, no crashed threads, total trade count matches DB record count
- Regression test sweep: run the full existing pytest suite after the refactor is complete and confirm nothing broke
- Round-based vs autonomous parity check: same seed, same config, assert final prices are within tolerance of each other

**Done when:** all existing Phase 1/2 tests still pass after the autonomous refactor, new integration tests cover the full stack (database → market service → API → autonomous agents → frontend), and a full test report is ready to include in the final deliverable write-up.

---

## Dependency Graph

```
Task 1 (DB) ────┐
                ├──→ Task 2 (Service) ──→ Task 3 (API) ──┬──→ Task 6 (Runner) ──┬──→ Task 7 (Frontend) ──→ Task 8 (Benchmarks) ──→ Task 10 (Integration)
                │                                        │                      │
Task 4 (Agent) ─┘                                        │                      │
                                                         │                      │
Task 5 (Personality) ────────────────────────────────────┘                      │
                                                                                │
Task 9 (Unit tests) ← baked into every task, owned by each dev ─────────────────┘
```

**Parallel start candidates (day 1):**
- Task 1 (DB schema)
- Task 4 (CRRA math extraction + agent skeleton)
- Task 5 (Personality dataclass)

**Blocked until prerequisites:**
- Task 2 waits on Task 1
- Task 3 waits on Task 2
- Task 6 waits on Task 3
- Task 7 can start with mocked endpoints in parallel with Task 6
- Task 8 waits on Task 6

---

## Assignment Suggestion (6 people)

| Person | Primary | Test Responsibility (Task 9) |
|---|---|---|
| Database Dev | Task 1 (DB layer) | Unit tests for DB layer |
| Market Service Dev | Task 2 (market service) | Unit tests for market service |
| API Lead | Task 3 (API) + Task 6 (Runner) | Unit tests for API + runner |
| Agent Dev | Task 4 (Agent client) + Task 5 (Personality) | Unit tests for agent + personality |
| Frontend | Task 7 | Component smoke tests |
| Coordinator | Task 8 + Task 10 (Integration Testing) | Full-stack integration + regression |

Task 9 (unit tests) is a shared requirement — each dev writes tests for the module they own. Task 10 (integration testing) is a standalone job owned by the Coordinator after the other tasks land.
