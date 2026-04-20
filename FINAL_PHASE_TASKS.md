# Final Phase: Autonomous Agent Architecture

Target: Working autonomous trading system for final presentation (~1 month out).

---

## Design Decisions (Locked)

| Question | Decision |
|---|---|
| Async model | Simulated async — threads in one Python process, not multi-process |
| Number of markets | Multi-market support is first-class (agents can participate in multiple markets concurrently) |
| Round-based mode | Kept alongside autonomous mode, both paths coexist |
| Persistence | SQLite (file-backed or `file::memory:?cache=shared`), WAL mode, shared across threads |
| HTTP client for agents | `requests` library (sync), not `aiohttp` |
| Concurrency | **SQLite WAL + `BEGIN IMMEDIATE` + `busy_timeout`** — per-thread connections in `MarketService`, no Python-level locks. Write serialization handled by SQLite's native locking. |
| CRRA math location | Extract shared helper to `src/crra_math.py` with `compute_optimal_trade(belief, price, cash, shares, rho) -> float`. Both `crra_agent.py` and `autonomous_agent.py` import from it |
| Agent lifecycle | **Decoupled from markets**: agents are created/updated globally, discover open markets, and choose where to trade without forced enrollment |

---

## Locked API Contracts

Every endpoint returns JSON. Errors return `{"error": "message"}` with appropriate HTTP status.

### POST /api/agents

**Request:**
```json
{
  "name": "agent_007",
  "cash": 100.0,
  "belief": 0.62,
  "rho": 1.0,
  "personality": { /* see personality schema below */ }
}
```

**Response (201):**
```json
{
  "agent_id": 7,
  "name": "agent_007",
  "cash": 100.0,
  "belief": 0.62,
  "rho": 1.0,
  "personality": { /* ... */ }
}
```

**Errors:** 400 for invalid belief/rho/personality values.

---

### GET /api/agents

**Response (200):**
```json
{
  "agents": [
    {"agent_id": 7, "name": "agent_007", "cash": 100.0, "belief": 0.62, "rho": 1.0, "personality": { /* ... */ }},
    {"agent_id": 8, "name": "agent_008", "cash": 120.0, "belief": 0.48, "rho": 0.8, "personality": { /* ... */ }}
  ],
  "total": 2
}
```

**Query params:** `?limit=100&offset=0`.

---

### PATCH /api/agents/{agent_id}

Updates global agent profile fields (`cash`, `belief`, `rho`, `personality`).

**Request (example):**
```json
{
  "belief": 0.70,
  "personality": { "edge_threshold": 0.02 }
}
```

**Response (200):** Updated agent object.

**Errors:** 404 if agent not found.

---

### POST /api/market/create

**Request:**
```json
{
  "mechanism": "lmsr",
  "title": "Will ETH close above $4k by Friday?",
  "ground_truth": 0.70,
  "b": 200.0
}
```

**Response (201):**
```json
{
  "market_id": "m_abc123",
  "mechanism": "lmsr",
  "initial_price": 0.5,
  "ground_truth": 0.70,
  "status": "open"
}
```

**Errors:** 400 if mechanism invalid or ground_truth out of [0.01, 0.99].

---

### GET /api/markets

Returns market discovery feed for autonomous agents and UI.

**Response (200):**
```json
{
  "markets": [
    {
      "market_id": "m_abc123",
      "title": "Will ETH close above $4k by Friday?",
      "mechanism": "lmsr",
      "status": "open",
      "price": 0.6847,
      "trade_count_24h": 142,
      "active_agents_24h": 31
    }
  ],
  "total": 1
}
```

**Query params:** `?status=open&limit=100&offset=0`.

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

This updates the **global agent belief**. Market positions are created lazily when the agent first trades in a market.

---

### GET /api/market/{market_id}/agents

**Response (200):** Array of agents with positions/trades in this market (same shape as single-agent endpoint).

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

Starts autonomous trading for this market. Global agent threads discover this market via `GET /api/markets` and may choose to trade it.

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
| Agent trades a market with no existing position row | Create position lazily on first trade, then apply trade atomically |
| Trade arrives for a stopped market | Return 409, agent client waits 1s and retries |
| Concurrent trades on same market | Serialized by market lock, later trade sees updated price |
| Agent thread crashes | Runner logs exception, restarts that agent after 2s |
| DB write fails mid-trade | Rollback transaction, return 500, no partial state |
| Belief shift while agent is mid-trade-computation | Agent re-reads belief at start of next cycle, no mid-cycle abort |
| Market stopped while agents running | Runner sets stop flag, agents exit cleanly within next poll cycle |

---

## Task 1 — Persistent Market State Layer  ✅

**File:** `app/market_store.py`

**Build:**
- `MarketStore` class with SQLite connection (in-memory or file-backed)
- Supports `_conn` parameter for external connection injection and `_external_transactions` flag so `MarketService` can manage transactions externally
- Thread safety provided by SQLite WAL mode + `busy_timeout` + `BEGIN IMMEDIATE` transactions (in the service layer), not per-market Python locks — avoids deadlocks and lock-ordering bugs
- Schema: `markets` (with `ground_truth`, `mechanism`, `status` lifecycle), `agents` (top-level, independent of markets — stores `belief`, `rho`, `cash`, `personality`), `positions` (join table linking agents to markets — stores market-specific holdings), `trades`, `orders`
  - **Key design: agents exist independently from markets.** An agent is created with its own beliefs/personality/cash first, then autonomously chooses markets. Markets do not create or configure agents.
- Methods:
  - `create_market(slug, title, mechanism, b, ground_truth, ...) -> dict`
  - `create_agent(name, cash, belief, rho, personality) -> dict` (no market_id — agent is independent)
  - `ensure_position(agent_id, market_id) -> dict` (lazy-create a position row when first needed)
  - `get_price(market_id) -> float`
  - `get_market(market_id) -> dict`
  - `get_agent(agent_id) -> dict` (agent-level data, not market-specific)
  - `get_position(agent_id, market_id) -> dict` (market-specific shares)
  - `update_agent_portfolio(market_id, agent_id, cash_delta, shares_delta) -> dict`
  - `set_agent_belief(agent_id, new_belief) -> old_belief` (belief is agent-level, not per-market)
  - `get_order_book(market_id) -> dict` (bids/asks)
  - `get_trades(market_id, since_trade_id=None, limit=100) -> list`
  - `set_market_status(market_id, status)` where status in {"created", "open", "running", "stopped"}
  - `submit_trade(agent_id, market_id, side, shares) -> dict` (LMSR)
  - `submit_limit_order / submit_market_order` (CDA)
  - `resolve_market(market_id, outcome) -> dict`

**Done when:**
- `pytest tests/test_market_store.py` passes (expanded): CRUD, LMSR/CDA trading, multi-market, resolution, status lifecycle, lazy position creation, agent belief/portfolio, `since_trade_id` filtering
- Decoupling invariants hold: creating a market does not create agents; creating an agent does not require a market; one agent can trade multiple markets

---

## Task 2 — Thread-Safe Market Service  ✅

**File:** `app/market_service.py`

**Build:**
- `MarketService` class that **depends on `MarketStore`** — creates per-thread `MarketStore` instances wrapping per-thread SQLite connections; delegates all CRUD and query methods to the store instead of re-implementing SQL
- Per-thread connections configured with WAL mode, `busy_timeout`, and `isolation_level=None` for explicit `BEGIN IMMEDIATE` transactions
- Add market-discovery helpers (`list_markets_with_summary`) and lazy-position helpers (`ensure_position`) used by trade execution paths
- `execute_lmsr_trade(market_id, agent_id, quantity) -> dict` — atomic under `BEGIN IMMEDIATE`: uses `LMSRMarketMaker` from `team_a_market_logic.py` for cost/price math, clips to affordable quantity when agent has insufficient cash
- `execute_cda_order(market_id, agent_id, side, quantity, limit_price, order_type) -> dict` — atomic: hydrates `ContinuousDoubleAuction` from `team_b_market_logic.py`, runs matching, persists results with per-trade cash validation
- `get_price_snapshot(market_id) -> dict` — returns mechanism-specific price info
- `get_price(market_id) -> float` — uses `LMSRMarketMaker` for LMSR, DB reference price for CDA
- Backward-compatible wrappers: `execute_trade`, `execute_limit_order`, `execute_market_order`

**Done when:**
- `pytest tests/test_market_service.py` passes (33 tests): 10 concurrent `execute_lmsr_trade` calls produce consistent final state, CDA order matching works with crossing orders, clipping works when agent has insufficient cash, `get_price_snapshot` returns correct dict
- First trade by an agent in a market auto-creates position state with no separate pre-linking call

---

## Task 3 — Agent-Facing API Endpoints

**File:** `api/main.py` (breaking redesign allowed)

**Build:**
- Import `MarketService` from `app/market_service.py` as a module-level singleton
- Add all endpoints listed in "Locked API Contracts" section above
- Market-create endpoints must not accept or infer agent population parameters
- Add top-level agent lifecycle endpoints: `POST /api/agents`, `GET /api/agents`, `PATCH /api/agents/{agent_id}`
- Do not add manual assignment endpoints; market participation is agent-driven via market discovery
- Add `GET /api/markets` — list all markets with current price, active agent count, trade count
- Keep/extend `POST /api/market/{id}/news` for information shocks to selected active agents
- Each endpoint validates input, calls service, returns dict (FastAPI handles JSON)
- Use Pydantic models for request validation — create `AgentCreateRequest`, `MarketCreateRequest`, `TradeRequest`, `BeliefUpdateRequest`, `NewsEventRequest` classes
- Response models optional but recommended

**Done when:**
- Manual curl flow works: create agents → create market → query markets → submit trade → verify new price → shift belief → verify
- Agents can be created before any market exists
- Same agent can trade multiple markets
- Market creation never creates agents or positions
- `GET /api/markets` returns list of all markets with summary info
- `POST /api/market/{id}/news` shifts beliefs for sensitive agents and price moves persistently (not just for a few rounds)
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
- `AutonomousAgent` class with `__init__(agent_id, api_base_url, personality, belief, rho, cash)`
  - Agent is created **without any market_id** — it exists independently first
  - Agent discovers open markets via `GET /api/markets` and chooses where to trade
  - No pre-assignment step
- `self._stop_flag = threading.Event()`
- `run()` method — loop until stop flag:
  1. Sleep `uniform(mean - jitter, mean + jitter)` seconds
  2. GET `/api/markets` → see all open markets
  3. Score candidate markets and pick one (start with biggest absolute edge `|belief - price|`; fallback random)
  4. GET `/market/{id}/price` → extract current price
  5. GET `/market/{id}/agent/{id}` → extract cash, shares, belief
  6. Compute `x_star = compute_optimal_trade(belief, price, cash, shares, rho)`
  7. Apply trade_size_noise: `x_star *= uniform(1 - noise, 1 + noise)`
  8. If `abs(belief - price) < edge_threshold`, skip
  9. If `random() > participation_rate`, skip
  10. POST `/market/{id}/trade` with quantity
  11. Log result (or error, with retry on 409)
- `stop()` sets the event flag
- Use `requests.Session()` for connection reuse
- Market selection strategy: start simple (round-robin), can be upgraded to edge-based later

**Done when:**
- `pytest tests/test_autonomous_agent.py` passes with mocked HTTP responses
- Manual test: start the API server with 2 open markets, run one `AutonomousAgent` for 10 seconds, observe it independently trade both markets
- `crra_agent.py` tests all still pass after math extraction

---

## Task 5 — Agent Personality System

**Files:** `src/personality.py`, modify `src/autonomous_agent.py`

**Build:**
- `Personality` dataclass with all 7 fields from schema, serializable via `asdict()`
- `sample_personality(distribution_config, rng) -> Personality` — samples each field from its configured distribution
- `DEFAULT_POPULATION_DIST` constant matching the spec above
- Wire `AutonomousAgent.run()` to use all personality fields (steps 1, 5, 6, 7 above)
- At **agent creation time** in Task 3, each agent gets a sampled personality stored in DB (or explicitly supplied)

**Done when:**
- Two agents with identical beliefs/cash/rho but different personalities demonstrably trade differently in a 30-second run (different trade counts, different average trade sizes)
- Unit tests cover all 7 personality fields affecting behavior
- Serialization round-trip works: `Personality -> dict -> JSON -> dict -> Personality`

---

## Task 6 — Multi-Threaded Agent Orchestrator

**File:** `src/agent_runner.py`

**Build:**
- `AgentRunner` class owned by the API layer (manages agents across all markets, not just one)
- `__init__(api_base_url, market_db)`
- `start_market(market_id)` — marks market active/open for autonomous trading and ensures global agent threads can discover it
- `stop_market(market_id)` — marks market stopped so agents skip it on next discovery cycle; updates market status in DB
- `start_all()` / `stop_all()` — convenience methods across all markets
- `is_running(market_id) -> bool`
- `agent_count_active(market_id) -> int`
- Thread-restart logic: if any agent thread dies unexpectedly, log and restart it (cap at 3 restarts per agent)
- Wire to `/api/market/{id}/start` and `/stop` endpoints
- Agents self-select markets each cycle; no market-to-agent assignment table is required

**Done when:**
- POST start on 2 markets → 50 agents trade across both for 30 seconds → POST stop on one market → agents on the other keep going → POST stop on second → all threads exit cleanly
- No zombie threads after stop
- Killing one agent mid-run triggers a clean restart, final state consistent

---

## Task 7 — Polymarket-Style Frontend

**Files:** `frontend/src/App.tsx`, new components under `frontend/src/components/`

The frontend should look and feel like a real prediction market platform (Polymarket), not a simulation dashboard. De-emphasize simulation parameters, emphasize the market experience.

**Build:**

**Market listing page (home):**
- Shows all open markets as cards: title, current price, agent count, trade volume
- "Create Market" button opens a market-only form (mechanism, title, ground_truth, liquidity params)
- Polls `GET /api/markets` every 2s to refresh

**Individual market page (`/market/{id}`):**
- Market title and current price prominently displayed (e.g. "Will it rain tomorrow? 68% YES")
- `LivePriceChart` — polls `/market/{id}/price` every 500ms
- `TradeFeed` — polls `/market/{id}/trades?since=<last_id>` every 1s, shows recent trades with agent name, side, quantity, price
- `OrderBook` (CDA only) — polls `/market/{id}/book` every 1s, shows bid/ask depth
- `CommentSection` — LLM-generated trader commentary per market, polls and appends new comments. Carry over existing Ollama integration from `api/llm_comments.py`
- Start/Stop trading buttons
- "News Event" button — opens a modal to trigger a news event (`POST /market/{id}/news`), input the new belief target value and what % of agents are affected

**Agent profiles page (`/agents`):**
- Table of all agents: name/id, belief, rho, cash, shares, PnL, personality traits
- Sortable columns, click into individual agent to see their trade history
- Manual belief shift control per agent

**Agent autonomy behavior (required):**
- No enroll/assign panel in the UI
- Show market discovery as automatic: when a market is open/running, autonomous agents can choose it
- Start button is not gated by manual assignment

**General:**
- Use `useEffect` + `setInterval` for polling, clean up on unmount
- Existing round-based mode remains accessible (e.g. under a "Classic Mode" tab) but is not the default
- Responsive layout

**Done when:**
- PM can open the app and it looks like a trading platform, not a simulation tool
- Full demo flow: create market with a title → start → autonomous agents discover and trade it → watch live price and trade feed → trigger a "news event" → see price react → read LLM comments → check agent leaderboard → stop → see final results
- Multiple markets can be open at the same time, agents trade across them
- LLM comments appear in each market's comment section
- Round mode still accessible and functional
- No console errors

---

## Task 8 — Autonomous Mode Evaluation

**Files:** `run_autonomous_benchmark.py`, `outputs/autonomous/*`

**Build:**

Four experiments, each exporting CSV + JSON + chart:

**Experiment 1 — Autonomous vs Round-Based Convergence**
- Same seed, same active global agent pool size (50), same ground_truth (0.70), same belief distribution
- Run autonomous for 60 seconds, run round-based for 100 rounds
- Compare: final price, final error, trajectory shape
- Success criterion: autonomous converges within 0.05 of round-based final price

**Experiment 2 — Personality Diversity Effect**
- Three configs: all agents identical, moderately varied, highly varied
- Run autonomous for 60 seconds each, 5 seeds per config
- Compare: final error, trade count, price stability (rolling std)
- Deliverable: 3-column table with mean ± std per metric

**Experiment 3 — News Event Response**
- Run autonomous for 30 seconds (baseline period)
- Trigger a "news event" that shifts beliefs of ~50% of agents (those with high signal_sensitivity) by +0.20
- Run another 30 seconds (recovery period)
- Measure: time for price to move 50% of the way to new mean belief, whether shift persists (unlike old round-based shocks that faded in a few rounds)
- Deliverable: overlay chart showing pre/post-news price trajectory

**Experiment 4 — Multi-Market**
- Create 2 markets with different ground truths (0.30 and 0.80)
- Let the same global agent pool discover and choose across both
- Run for 60 seconds, verify each market converges toward its own ground truth independently
- Deliverable: side-by-side price charts

**Done when:**
- All four experiments run end-to-end with one command per experiment
- Output artifacts exist in `outputs/autonomous/` (CSVs, JSON, matplotlib PNGs)
- Summary table ready to drop into the final deliverable and slides

---

## Task 9 — Unit Test Coverage for New Modules (Distributed)

Each developer writes unit tests for the module they own as part of their task. This is not a separate assignment — it's a requirement baked into Tasks 1 through 7.

**Per-task test files:**
- Task 1 owner → `tests/test_market_store.py`
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
- Decoupling invariant test: create market, assert no agents/positions are auto-created
- Belief shock integration test: start market, run 10 seconds, inject news/shift, run another 10 seconds, assert price moved toward new belief mean
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
