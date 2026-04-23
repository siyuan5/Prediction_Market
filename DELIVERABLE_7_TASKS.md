# Deliverable 7: Final Polish + New Features

Last deliverable before the May 11 presentation. Focus is docs and small features, plus cleanup of existing bugs. Nothing here is a big architectural change.

---

## Priorities (what matters most)

1. Documentation — this is the top priority. Final presentation + external reviewers
2. Resolution + settlement button on market page
3. Agent detail page with trades/comments/PnL
4. Delete agents
5. Persist graph data across page reloads
6. News event history per market
7. General bug fixes (start/stop flakiness is the known one, there are probably others)
8. Expose comments via API if not already
9. Stretch: LLM comments influence other traders (skip if time is tight)

---

## Design Decisions (Locked)

| Question | Decision |
|---|---|
| Graph data persistence | Use existing trades table. On page load, frontend fetches last N trades and plots their price_after. No new tables or timeseries storage needed |
| Comments influencing traders | Same pattern as news events. Agents with high signal_sensitivity read recent comments and blend the commenter's belief into their own via signal_sensitivity × (1 - stubbornness). No new plumbing |
| Comments API shape | Follow existing /api/market/{id}/comments that's already wired up. Add DELETE if missing. Don't rewrite it |
| Delete agent scope | Full cascade: delete agent, positions, orders, and trades. Or soft-delete with a deleted_at column. Pick one per cost, soft-delete is safer for the demo |
| Market resolution | Uses existing svc.resolve_market() in market_store. Just needs UI button + endpoint wrapper |

---

## Task 1 — Documentation Pass

**Owner:** Coordinator + everyone reviews their own module

**Build:**
- Rewrite the top-level README.md. Should cover: what the project is, quick start (install + run), architecture overview in 1 paragraph per layer, how to run benchmarks, how to run tests
- Add a short ARCHITECTURE.md with a diagram (can be ASCII) showing frontend → API → market service → DB + agent runner
- Every top-level module in src/ and app/ gets a module docstring at the top describing what it does
- Any API endpoint missing a description in the FastAPI router should get one
- Clean up any TODO comments left in the code. Either delete or convert to proper follow-up notes

**Done when:**
- A new reader can clone the repo, read the README, and get the frontend running in under 10 min
- ARCHITECTURE.md exists and is accurate
- All modules in src/ and app/ have a docstring at the top

---

## Task 2 — Market Resolution and Settlement

**Files:** `api/market_routes.py`, `frontend/src/pages/MarketDetailPage.tsx`

**Build:**
- Add POST /api/market/{id}/resolve that takes `{outcome: "yes" | "no"}` and calls svc.resolve_market(market_id, outcome)
- On success, returns the settlement dict (winners, losers, total payout)
- Market status moves to "resolved" in the DB. Trading rejected after that
- Frontend: button on market detail page labeled "Resolve market". Opens a small modal to pick Yes or No, confirms, shows settlement results inline
- Resolved markets should display their outcome and final payout summary on the market page instead of the trade controls

**Done when:**
- Can resolve a market from the UI and see PnL distributed to agents
- Agent cash values update correctly after resolution
- Cannot trade on a resolved market (endpoint returns 409 or 400)

---

## Task 3 — Agent Detail Page

**Files:** `frontend/src/pages/AgentDetailPage.tsx` (new), `frontend/src/App.tsx` (routing), possibly new API endpoints

**Build:**
- New route `/agents/:agentId` that shows one agent in detail
- Page sections:
  - Header: name, current cash, rho, personality summary, total PnL across all markets
  - Markets joined: table with market title, position (shares held), unrealized PnL, link to market
  - Trade history: list of every trade the agent made, sortable by time, with market title + price + quantity
  - Comments made: list of every comment the agent has posted
- Clickable rows on the Agents roster page link here
- API: if no existing endpoint returns cross-market trades for one agent, add GET /api/agents/{id}/trades
- Similarly GET /api/agents/{id}/comments if needed

**Done when:**
- Click on any agent in the roster → detail page loads with all their activity
- PnL number matches what the market pages show for that agent

---

## Task 4 — Delete Agents

**Files:** `app/market_store.py`, `app/market_service.py`, `api/market_routes.py`, frontend

**Build:**
- Soft delete is the safer choice. Add `deleted_at` column to agents table. Store filters it out on list/get by default
- API: DELETE /api/agents/{id}. Returns 200 with `{status: "deleted"}`
- Frontend: small delete button next to each agent on the roster page with a confirm dialog
- If the agent is currently being traded by the runner, stop their thread first (check agent_runner)
- Deleted agents should not show up on market detail pages, agent listings, or trade feeds

**Done when:**
- Can delete an agent from the UI
- They stop appearing in any agent-facing view
- Their past trades still exist in the DB (not cascaded) so market history stays intact

---

## Task 5 — Persistent Graph Data

**Files:** `frontend/src/pages/MarketDetailPage.tsx`

**Build:**
- On page load, fetch GET /api/market/{id}/trades?limit=500 and plot `price_after` against `created_at` for each trade row
- This gives a real historical price chart that survives page reloads
- After that, live polling appends new trades on top as they come in (that already works)
- No backend changes, just use trades data instead of starting from scratch

**Done when:**
- Refresh the market page mid-trading → chart still shows the full history, doesn't reset to flat

---

## Task 6 — News Event History Panel

**Files:** `app/market_store.py` (possibly new table), `api/market_routes.py`, frontend

**Build:**
- If there isn't already a news_events table, add one: id, market_id, headline, target_belief, affected_count, created_at
- POST /api/market/{id}/news should insert a row in this table each time it fires
- Add GET /api/market/{id}/news that returns past events for that market
- Frontend: new section on market detail page, collapsible, shows past events with timestamp, headline, and how many agents were affected
- Also mark each news event with a vertical line on the price chart at its timestamp

**Done when:**
- Every news event that's triggered persists and shows up in the history section
- Refreshing the page still shows past events
- Markers appear on the price chart at the right times

---

## Task 7 — Bug Fixes

**Files:** varies

**Known issues to fix:**
- Start/stop trading sometimes doesn't work properly. Repro: start → stop → start again, second start may not spawn agents. Check agent_runner for state not being fully reset
- Anything else teammates flag during testing

**General cleanup:**
- Run the full test suite, fix any flakes
- Clean up console errors in the frontend if any
- Make sure the frontend handles a market with zero trades without crashing

**Done when:**
- Start/stop works consistently across multiple cycles
- `pytest tests/` passes clean, no warnings
- No red errors in browser console during the demo flow

---

## Task 8 — Comments API Audit + LLM Comments Influence Traders (Stretch)

**Files:** `api/market_routes.py`, `src/autonomous_agent.py`

**Build:**

Part A (audit):
- Check that /api/market/{id}/comments GET returns full comment history with pagination
- Add DELETE /api/market/{id}/comments/{comment_id} if teammates want moderation
- Make sure comments have `agent_id`, `belief`, `at`, `source` (llm vs template), `text`

Part B (stretch, only if time):
- In AutonomousAgent.run_cycle, after picking a market, fetch last ~5 comments from that market
- For each comment, treat the commenter's belief as an external signal
- Apply the same signal_sensitivity × (1 - stubbornness) blend used for news events
- Effect: high-sensitivity agents nudge their belief toward the commenter's belief

**Done when:**
- Comments API has full CRUD coverage (or at least list + delete)
- (Stretch) An agent with high signal_sensitivity visibly drifts when repeatedly shown comments from a bullish commenter

---

## Task 9 — Final Deliverable Write-up

**Owner:** Coordinator

**Build:**
- Draft Deliverable 7 document matching the D5/D6 format
- Cover each new feature with a short implementation detail section
- Include a "final system overview" section since this is the last deliverable before presentation
- Update the repo structure and file list one last time

**Done when:**
- Deliverable 7 is submitted on time
- Ready to go into presentation prep

---

## Dependency Graph

```
Task 1 (Docs)         ← can start anytime, continuous
Task 2 (Resolution)   ← backend + frontend, independent
Task 3 (Agent Detail) ← needs Task 4 considered for deleted-agent handling
Task 4 (Delete)       ← backend + frontend, independent
Task 5 (Graph)        ← frontend only, independent
Task 6 (News history) ← backend (new table) + frontend
Task 7 (Bugs)         ← whoever finds them
Task 8 (Comments)     ← independent, stretch
Task 9 (Write-up)     ← end of sprint
```

Parallel day 1 candidates: Tasks 1, 2, 4, 5 can all start immediately.

---

## Suggested Assignment (6 people)

| Person | Tasks |
|---|---|
| Person 1 | Task 2 (resolution + settlement) + Task 7 (start/stop bug fix) |
| Person 2 | Task 4 (delete agents) + Task 6 (news event history) |
| Person 3 | Task 3 (agent detail page) |
| Person 4 | Task 5 (persistent graph) + Task 8 (comments API + stretch) |
| Person 5 | general bug hunt (Task 7) + help wherever needed |
| Coordinator | Task 1 (docs) + Task 9 (final write-up) |

Bug hunt is shared by whoever finds things during testing, not just Person 5. Everyone should run through the demo flow before submission and log anything broken.
