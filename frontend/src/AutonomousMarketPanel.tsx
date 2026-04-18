/**
 * Live “autonomous market” UI: SQLite-backed LMSR market exposed at ``/api/market/*``.
 *
 * Flow:
 *   1. **Create agents + market** — create global agents (POST /api/agents/create),
 *      then create a market (POST /api/market/create).
 *   2. **Start** — server spawns autonomous trader threads that poll and trade (POST .../start).
 *   3. **Polling** — this component repeatedly fetches price and new trades (GET .../price, .../trades).
 *   4. **Belief shock** — optional POST to nudge one agent’s belief mid-run (belief shock experiment).
 *   5. **Stop** — stops threads and sets market status to stopped (POST .../stop).
 *
 * The Vite dev server proxies ``/api`` to the FastAPI backend (see vite.config.ts).
 */

import { useCallback, useEffect, useRef, useState } from "react";

type PriceSnapshot = {
  market_id: number;
  price: number;
  best_bid: number | null;
  best_ask: number | null;
  last_trade_price?: number | null;
  last_trade_at?: string | null;
  timestamp: string;
};

type TradeRow = {
  trade_id: string;
  agent_id: number;
  quantity: number;
  price: number;
  at?: string;
};

type CreateResponse = {
  market_id: number;
  mechanism: string;
  initial_price: number;
  ground_truth: number;
};

export function AutonomousMarketPanel() {
  const [nAgents, setNAgents] = useState(12);
  const [b, setB] = useState(100);
  const [groundTruth, setGroundTruth] = useState(0.65);
  const [seed, setSeed] = useState(42);
  const [title, setTitle] = useState("Live LMSR (autonomous)");

  const [marketId, setMarketId] = useState<number | null>(null);
  const [createInfo, setCreateInfo] = useState<CreateResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [price, setPrice] = useState<PriceSnapshot | null>(null);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  /** Latest SQLite trade id we have appended (for ``since=`` incremental fetch). */
  const lastTradeIdRef = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [shockAgent, setShockAgent] = useState(1);
  const [shockBelief, setShockBelief] = useState(0.55);

  /** Incremented when user clicks Stop so polling effect tears down cleanly. */
  const pollGeneration = useRef(0);

  const fetchPrice = useCallback(async (mid: number) => {
    const res = await fetch(`/api/market/${mid}/price`);
    if (!res.ok) throw new Error(await res.text());
    setPrice((await res.json()) as PriceSnapshot);
  }, []);

  /**
   * Pull trades with `since` = last seen trade id so the feed only appends new rows.
   * Falls back to full list on first poll (lastTradeId === 0).
   */
  const fetchTrades = useCallback(async (mid: number) => {
    const since = lastTradeIdRef.current;
    const q = since > 0 ? `?since=${since}&limit=80` : "?limit=80";
    const res = await fetch(`/api/market/${mid}/trades${q}`);
    if (!res.ok) throw new Error(await res.text());
    const data = (await res.json()) as { trades: TradeRow[] };
    const batch = data.trades ?? [];
    if (batch.length === 0) return;
    setTrades((prev) => {
      const merged = [...prev, ...batch];
      return merged.slice(-200);
    });
    const maxId = Math.max(
      since,
      ...batch.map((t) => Number.parseInt(t.trade_id, 10) || 0),
    );
    lastTradeIdRef.current = maxId;
  }, []);

  /** While ``marketId`` is set and ``pollKey`` matches, poll price + trades on an interval. */
  useEffect(() => {
    if (marketId == null) return;
    const gen = pollGeneration.current;
    let cancelled = false;
    const tick = async () => {
      if (cancelled || pollGeneration.current !== gen) return;
      try {
        await fetchPrice(marketId);
        await fetchTrades(marketId);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void tick();
    const id = window.setInterval(tick, 900);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [marketId, fetchPrice, fetchTrades]);

  async function handleCreate() {
    setBusy(true);
    setError(null);
    setTrades([]);
    lastTradeIdRef.current = 0;
    setCreateInfo(null);
    setPrice(null);
    try {
      const runTag = Date.now();
      for (let i = 0; i < nAgents; i += 1) {
        const beliefNoise = (Math.random() - 0.5) * 0.2;
        const belief = Math.min(0.99, Math.max(0.01, groundTruth + beliefNoise));
        const agentRes = await fetch("/api/agents/create", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: `ui-auto-${runTag}-${i}`,
            cash: 100.0,
            belief,
            rho: 1.0,
            personality: {
              check_interval_mean: 2.0,
              check_interval_jitter: 1.0,
              edge_threshold: 0.03,
              participation_rate: 0.8,
              trade_size_noise: 0.2,
              signal_sensitivity: 0.5,
              stubbornness: 0.3,
            },
          }),
        });
        if (!agentRes.ok) throw new Error(await agentRes.text());
      }

      const res = await fetch("/api/market/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mechanism: "lmsr",
          ground_truth: groundTruth,
          b,
          title,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as CreateResponse;
      setCreateInfo(data);
      setMarketId(data.market_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleStart() {
    if (marketId == null) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/market/${marketId}/start`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      setRunning(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleStop() {
    if (marketId == null) return;
    pollGeneration.current += 1;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/market/${marketId}/stop`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      setRunning(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function handleBeliefShock() {
    if (marketId == null) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/market/${marketId}/agent/${shockAgent}/belief`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_belief: shockBelief }),
      });
      if (!res.ok) throw new Error(await res.text());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel autonomous-panel" aria-labelledby="autonomous-heading">
      <h2 id="autonomous-heading">Autonomous trading (live)</h2>
      <p className="sub autonomous-sub">
        Creates a <strong>persistent LMSR</strong> market in SQLite, then optionally starts background
        traders that call the same HTTP API. Use this section to watch <strong>price updates</strong>, a{" "}
        <strong>trade feed</strong>, and <strong>belief shocks</strong> on individual agents. Requires the
        API running (e.g. <code>uvicorn api.main:app</code>).
      </p>

      <div className="form-grid">
        <label>
          Title
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            disabled={busy || marketId != null}
          />
        </label>
        <label>
          Agents
          <input
            type="number"
            min={2}
            max={200}
            value={nAgents}
            onChange={(e) => setNAgents(Number(e.target.value))}
            disabled={busy || marketId != null}
          />
        </label>
        <label>
          LMSR b
          <input
            type="number"
            min={1}
            step={1}
            value={b}
            onChange={(e) => setB(Number(e.target.value))}
            disabled={busy || marketId != null}
          />
        </label>
        <label>
          Ground truth
          <input
            type="number"
            step={0.01}
            min={0.01}
            max={0.99}
            value={groundTruth}
            onChange={(e) => setGroundTruth(Number(e.target.value))}
            disabled={busy || marketId != null}
          />
        </label>
        <label>
          Seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            disabled={busy || marketId != null}
          />
        </label>
        <div className="autonomous-actions">
          <button type="button" className="primary" disabled={busy || marketId != null} onClick={handleCreate}>
            {busy && marketId == null ? "Creating…" : "Create market"}
          </button>
          <button type="button" disabled={busy || marketId == null || running} onClick={handleStart}>
            Start autonomous traders
          </button>
          <button type="button" disabled={busy || marketId == null || !running} onClick={handleStop}>
            Stop
          </button>
        </div>
      </div>

      {error ? (
        <p className="autonomous-error" role="alert">
          {error}
        </p>
      ) : null}

      {createInfo ? (
        <div className="autonomous-status">
          <p>
            Market <code>#{createInfo.market_id}</code> · {nAgents} agents · P* ={" "}
            {createInfo.ground_truth.toFixed(2)} · initial price {createInfo.initial_price.toFixed(4)}
          </p>
          {price ? (
            <p className="autonomous-price">
              Live price (implied Yes): <strong>{(price.price * 100).toFixed(2)}%</strong>
              <span className="autonomous-ts"> · {price.timestamp}</span>
            </p>
          ) : (
            <p className="muted">Waiting for first poll…</p>
          )}
        </div>
      ) : null}

      <div className="autonomous-grid">
        <div>
          <h3>Trade feed</h3>
          <p className="muted small">
            Newest at bottom. Trades appear once autonomous agents run or you trade via API.
          </p>
          <div className="autonomous-feed" aria-live="polite">
            {trades.length === 0 ? (
              <span className="muted">No trades yet.</span>
            ) : (
              trades.map((t) => (
                <div key={`${t.trade_id}-${t.at}`} className="autonomous-feed-row">
                  <span className="mono">#{t.trade_id}</span> agent {t.agent_id} · qty {t.quantity.toFixed(3)} ·{" "}
                  {(t.price * 100).toFixed(2)}¢
                </div>
              ))
            )}
          </div>
        </div>
        <div>
          <h3>Belief shock (single agent)</h3>
          <p className="muted small">
            Sets that agent&apos;s belief directly (used to study price reaction). Agent ids are 1…N in
            creation order for most SQLite seeds.
          </p>
          <div className="form-grid compact">
            <label>
              Agent id
              <input
                type="number"
                min={1}
                step={1}
                value={shockAgent}
                onChange={(e) => setShockAgent(Number(e.target.value))}
                disabled={marketId == null || busy}
              />
            </label>
            <label>
              New belief
              <input
                type="number"
                step={0.01}
                min={0.01}
                max={0.99}
                value={shockBelief}
                onChange={(e) => setShockBelief(Number(e.target.value))}
                disabled={marketId == null || busy}
              />
            </label>
            <button type="button" disabled={busy || marketId == null} onClick={handleBeliefShock}>
              Apply shock
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
