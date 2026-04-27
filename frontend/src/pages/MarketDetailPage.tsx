import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Detail = {
  market_id: number;
  title: string;
  mechanism: string;
  /** open | running | stopped | … */
  status: string;
  resolution?: "yes" | "no" | null;
  resolved_at?: string | null;
  ground_truth: number;
  price: number;
  trade_count: number;
  /** Distinct agents who have traded in this market (not “spawned” count). */
  active_agents: number;
};

type PriceSnap = {
  market_id: number;
  price: number;
  best_bid: number | null;
  best_ask: number | null;
  last_trade_price?: number | null;
  timestamp: string;
  /** Hidden P(Yes) for this market (scenario). */
  ground_truth?: number | null;
  /** Mean belief across agents participating in this market. */
  mean_belief?: number | null;
};

type TradeRow = {
  trade_id: string;
  agent_id: number;
  quantity: number;
  price: number;
  at?: string;
};

type CommentRow = {
  id: number;
  agent_id: number;
  text: string;
  source?: string;
  at?: string;
};

type NewsEventRow = {
  id: number;
  market_id: number;
  headline: string;
  mode: string;
  requested_new_belief?: number | null;
  requested_delta?: number | null;
  n_affected: number;
  mean_belief_before: number;
  mean_belief_after: number;
  at_timestamp: string;
};

type SettlementRow = {
  agent_id: number;
  name: string;
  yes_shares: number;
  payout: number;
  cash_after: number;
};

type SettlementResult = {
  market_id: number;
  outcome: "yes" | "no";
  payoff_per_yes_share: number;
  positions_settled: number;
  total_payout: number;
  winners: SettlementRow[];
  losers: SettlementRow[];
  resolved_at: string;
};

export function MarketDetailPage() {
  const navigate = useNavigate();
  const { marketId: midParam } = useParams();
  const marketId = Number.parseInt(midParam ?? "", 10);

  const [detail, setDetail] = useState<Detail | null>(null);
  const [price, setPrice] = useState<PriceSnap | null>(null);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [book, setBook] = useState<{ bids: { price: number; quantity: number }[]; asks: { price: number; quantity: number }[] } | null>(null);
  const [comments, setComments] = useState<CommentRow[]>([]);
  const [chartRows, setChartRows] = useState<
    { t: number; mid: number; meanBelief: number | null; groundTruth: number | null }[]
  >([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const lastTradeIdRef = useRef(0);
  const lastCommentIdRef = useRef(0);
  const tickRef = useRef(0);
  const [demoN, setDemoN] = useState(12);
  const [showNews, setShowNews] = useState(false);
  const [newsHeadline, setNewsHeadline] = useState("Breaking: major headline shifts sentiment");
  const [newsBelief, setNewsBelief] = useState(0.75);
  const [newsFraction, setNewsFraction] = useState(0.5);
  const [newsMinSens, setNewsMinSens] = useState(0.5);
  const [globalAgentTotal, setGlobalAgentTotal] = useState<number | null>(null);
  const [spawnNotice, setSpawnNotice] = useState<string | null>(null);
  /** From GET /trades ``total`` — authoritative count of rows in DB for this market. */
  const [serverTradeTotal, setServerTradeTotal] = useState<number | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [newsHistory, setNewsHistory] = useState<NewsEventRow[]>([]);
  const [showResolve, setShowResolve] = useState(false);
  const [resolveBusy, setResolveBusy] = useState(false);
  const [settlement, setSettlement] = useState<SettlementResult | null>(null);

  const loadDetail = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    try {
      const res = await fetch(`/api/market/${marketId}/detail`);
      if (!res.ok) throw new Error(await res.text());
      const d = (await res.json()) as Detail;
      setDetail(d);
      setRunning(d.status === "running");
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [marketId]);

  const refreshGlobalAgentCount = useCallback(async () => {
    try {
      const res = await fetch("/api/agents?limit=1&offset=0");
      if (!res.ok) return;
      const j = (await res.json()) as { total: number };
      setGlobalAgentTotal(typeof j.total === "number" ? j.total : null);
    } catch {
      /* ignore */
    }
  }, []);

  const fetchPrice = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    const res = await fetch(`/api/market/${marketId}/price`);
    if (!res.ok) throw new Error(await res.text());
    const p = (await res.json()) as PriceSnap;
    setPrice(p);
    const t = Date.now();
    const mid = Number(p.price);
    const mb =
      p.mean_belief != null && Number.isFinite(Number(p.mean_belief)) ? Number(p.mean_belief) : null;
    const gt =
      p.ground_truth != null && Number.isFinite(Number(p.ground_truth)) ? Number(p.ground_truth) : null;
    setChartRows((rows) => {
      const next = [...rows, { t, mid, meanBelief: mb, groundTruth: gt }];
      return next.slice(-240);
    });
  }, [marketId]);

  const fetchTrades = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    const since = lastTradeIdRef.current;
    const q = since > 0 ? `?since=${since}&limit=80` : "?limit=80";
    const res = await fetch(`/api/market/${marketId}/trades${q}`);
    if (!res.ok) throw new Error(await res.text());
    const data = (await res.json()) as { trades: TradeRow[]; total?: number };
    if (typeof data.total === "number") {
      setServerTradeTotal(data.total);
    }
    const batch = data.trades ?? [];
    if (batch.length === 0) return;
    setTrades((prev) => [...prev, ...batch].slice(-200));
    const maxId = Math.max(since, ...batch.map((t) => Number.parseInt(t.trade_id, 10) || 0));
    lastTradeIdRef.current = maxId;
  }, [marketId]);

  const fetchBook = useCallback(async () => {
    if (!Number.isFinite(marketId) || detail?.mechanism !== "cda") return;
    const res = await fetch(`/api/market/${marketId}/book`);
    if (res.status === 400) {
      setBook(null);
      return;
    }
    if (!res.ok) return;
    setBook((await res.json()) as typeof book);
  }, [marketId, detail?.mechanism]);

  const fetchComments = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    const since = lastCommentIdRef.current;
    const res = await fetch(`/api/market/${marketId}/comments?since=${since}`);
    if (!res.ok) return;
    const data = (await res.json()) as { comments: CommentRow[] };
    const batch = data.comments ?? [];
    if (batch.length === 0) return;
    setComments((prev) => [...prev, ...batch].slice(-120));
    lastCommentIdRef.current = Math.max(since, ...batch.map((c) => c.id));
  }, [marketId]);

  const fetchNewsHistory = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    const res = await fetch(`/api/market/${marketId}/news?limit=200&offset=0`);
    if (!res.ok) return;
    const data = (await res.json()) as { events: NewsEventRow[] };
    setNewsHistory(data.events ?? []);
  }, [marketId]);

  const tickComments = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    await fetch(`/api/market/${marketId}/comments/tick`, { method: "POST" });
    await fetchComments();
  }, [marketId, fetchComments]);

  useEffect(() => {
    void loadDetail();
    void refreshGlobalAgentCount();
    const id = window.setInterval(loadDetail, 3000);
    return () => window.clearInterval(id);
  }, [loadDetail, refreshGlobalAgentCount]);

  useEffect(() => {
    setChartRows([]);
    lastTradeIdRef.current = 0;
    lastCommentIdRef.current = 0;
  }, [marketId]);

  useEffect(() => {
    if (!Number.isFinite(marketId)) return;
    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      try {
        await fetchPrice();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void run();
    const id = window.setInterval(run, 500);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [marketId, fetchPrice]);

  useEffect(() => {
    if (!Number.isFinite(marketId)) return;
    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      try {
        await fetchTrades();
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    void run();
    const id = window.setInterval(run, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [marketId, fetchTrades]);

  useEffect(() => {
    if (!Number.isFinite(marketId) || detail?.mechanism !== "cda") return;
    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      try {
        await fetchBook();
      } catch {
        setBook(null);
      }
    };
    void run();
    const id = window.setInterval(run, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [marketId, detail?.mechanism, fetchBook]);

  useEffect(() => {
    if (!Number.isFinite(marketId)) return;
    tickRef.current += 1;
    const gen = tickRef.current;
    let cancelled = false;
    const poll = async () => {
      if (cancelled || tickRef.current !== gen) return;
      try {
        await fetchComments();
      } catch {
        /* ignore */
      }
    };
    void poll();
    const id = window.setInterval(poll, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [marketId, fetchComments]);

  useEffect(() => {
    if (!Number.isFinite(marketId)) return;
    const gen = tickRef.current;
    let cancelled = false;
    const run = async () => {
      if (cancelled || tickRef.current !== gen) return;
      try {
        await tickComments();
      } catch {
        /* ignore */
      }
    };
    const id = window.setInterval(run, 3500);
    void run();
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [marketId, tickComments]);

  useEffect(() => {
    if (!Number.isFinite(marketId)) return;
    let cancelled = false;
    const run = async () => {
      if (cancelled) return;
      try {
        await fetchNewsHistory();
      } catch {
        /* ignore */
      }
    };
    void run();
    const id = window.setInterval(run, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [marketId, fetchNewsHistory]);

  async function handleStart() {
    setBusy(true);
    setError(null);
    setSpawnNotice(null);
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

  async function handleDeleteMarket() {
    if (!detail) return;
    const label = detail.title || `Market #${marketId}`;
    if (
      !window.confirm(
        `Delete "${label}"? This removes the market and its history. ` +
          `Traders who only participated here are removed from the pool; others are kept.`,
      )
    ) {
      return;
    }
    setDeleteBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/market/${marketId}`, {
        method: "DELETE",
        cache: "no-store",
        headers: { "Cache-Control": "no-cache" },
      });
      if (!res.ok) throw new Error(await res.text());
      navigate("/", { replace: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleteBusy(false);
    }
  }

  async function handleStop() {
    tickRef.current += 1;
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

  async function spawnDemoAgents() {
    if (!Number.isFinite(marketId)) return;
    if (!detail) {
      setError("Market is still loading — wait a second and try again.");
      return;
    }
    setBusy(true);
    setError(null);
    setSpawnNotice(null);
    try {
      const tag = Date.now();
      for (let i = 0; i < demoN; i += 1) {
        // Sample heterogeneous risk aversion so spawned agents are not identical.
        const rho = 0.5 + Math.random() * 1.5;
        const res = await fetch("/api/agents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: `trader-${tag}-${i}`,
            cash: 100.0,
            rho: Math.round(rho * 100) / 100,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
      }
      await loadDetail();
      await refreshGlobalAgentCount();
      setSpawnNotice(String(demoN));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function postNews() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/market/${marketId}/news`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          headline: newsHeadline,
          new_belief: newsBelief,
          affected_fraction: newsFraction,
          min_signal_sensitivity: newsMinSens,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setShowNews(false);
      await fetchPrice();
      await fetchNewsHistory();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function resolveMarket() {
    setResolveBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/market/${marketId}/resolve`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(await res.text());
      const payload = (await res.json()) as SettlementResult;
      setSettlement(payload);
      setRunning(false);
      setShowResolve(false);
      tickRef.current += 1;
      await loadDetail();
      await fetchPrice();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setResolveBusy(false);
    }
  }

  const chartData = useMemo(() => {
    if (chartRows.length === 0) return [];
    const t0 = chartRows[0].t;
    return chartRows.map((r, i) => ({
      i,
      sec: Math.round((r.t - t0) / 100) / 10,
      mid: Math.round(r.mid * 1000) / 10,
      meanBelief: r.meanBelief != null ? Math.round(r.meanBelief * 1000) / 10 : null,
      pStar: r.groundTruth != null ? Math.round(r.groundTruth * 1000) / 10 : null,
    }));
  }, [chartRows]);

  const newsMarkers = useMemo(() => {
    if (chartRows.length === 0 || newsHistory.length === 0) return [];
    const t0 = chartRows[0].t;
    return newsHistory
      .map((ev) => {
        const ts = Date.parse(ev.at_timestamp);
        if (!Number.isFinite(ts)) return null;
        const sec = Math.round(((ts - t0) / 1000) * 10) / 10;
        return { id: ev.id, sec, headline: ev.headline };
      })
      .filter((x): x is { id: number; sec: number; headline: string } => x != null)
      .filter((x) => x.sec >= 0);
  }, [chartRows, newsHistory]);

  if (!Number.isFinite(marketId)) {
    return (
      <p className="pm-error">
        Invalid market. <Link to="/">Back</Link>
      </p>
    );
  }

  const midPct = price ? Math.round(Number(price.price) * 1000) / 10 : null;
  const lastPrintPct =
    price?.last_trade_price != null && Number.isFinite(Number(price.last_trade_price))
      ? Math.round(Number(price.last_trade_price) * 1000) / 10
      : null;

  return (
    <div className="pm-page pm-detail">
      <nav className="pm-breadcrumb">
        <Link to="/">Markets</Link>
        <span className="pm-bc-sep">/</span>
        <span>{detail?.title ?? `Market #${marketId}`}</span>
      </nav>

      {error ? (
        <p className="pm-error" role="alert">
          {error}
        </p>
      ) : null}

      <header className="pm-detail-head">
        <div>
          <h1 className="pm-detail-title">{detail?.title ?? "Loading…"}</h1>
          <p className="pm-muted">
            {detail?.mechanism?.toUpperCase()} · P* {(detail?.ground_truth ?? 0) * 100}% · {detail?.trade_count ?? 0}{" "}
            trades · {detail?.active_agents ?? 0} traders with prints here
            {globalAgentTotal != null ? ` · ${globalAgentTotal} agents globally` : ""}
          </p>
          {detail ? (
            <button
              type="button"
              className="pm-btn-danger pm-detail-delete"
              disabled={busy || deleteBusy}
              onClick={() => void handleDeleteMarket()}
            >
              {deleteBusy ? "Deleting…" : "Delete market"}
            </button>
          ) : null}
        </div>
        <div className="pm-detail-yes-block" aria-live="polite">
          <div className="pm-yes-huge-label">Implied Yes (LMSR mid)</div>
          <div className="pm-yes-huge">{midPct != null ? `${midPct}%` : "—"}</div>
          {lastPrintPct != null ? (
            <div className="pm-ts">Last trade print: {lastPrintPct}%</div>
          ) : null}
          {price?.timestamp ? <div className="pm-ts">{price.timestamp}</div> : null}
        </div>
      </header>

      {spawnNotice ? (
        <div className="pm-spawn-notice" role="status">
          <p style={{ margin: 0 }}>
            Created {spawnNotice} agents in the global pool. Trade count / “traders with prints” only update after actual
            trades. <strong>Click Start trading</strong> below if you haven&apos;t — the chart stays flat until the
            market is running.
          </p>
        </div>
      ) : null}

      {detail?.status === "resolved" ? (
        <section className="pm-panel pm-controls-panel">
          <h2 className="pm-controls-title">Market resolved</h2>
          <p className="pm-muted small" style={{ marginTop: 0 }}>
            Outcome: <strong>{(settlement?.outcome ?? detail.resolution ?? "unknown").toUpperCase()}</strong>
            {detail.resolved_at ? ` · ${new Date(detail.resolved_at).toLocaleString()}` : ""}
          </p>
          <p className="pm-muted small">
            Positions settled: <strong>{settlement?.positions_settled ?? "n/a"}</strong> · Total payout:{" "}
            <strong>
              {typeof settlement?.total_payout === "number" ? settlement.total_payout.toFixed(2) : "n/a"}
            </strong>
          </p>
          {settlement ? (
            <div className="pm-two-col">
              <section className="pm-panel">
                <h3>Top payouts</h3>
                <div className="pm-feed">
                  {settlement.winners.length === 0 ? (
                    <span className="pm-muted">No settled traders.</span>
                  ) : (
                    settlement.winners.slice(0, 5).map((row) => (
                      <div key={`w-${row.agent_id}`} className="pm-feed-row">
                        Agent {row.agent_id} ({row.name}) · {row.yes_shares.toFixed(2)} shares · payout{" "}
                        {row.payout.toFixed(2)}
                      </div>
                    ))
                  )}
                </div>
              </section>
              <section className="pm-panel">
                <h3>Lowest payouts</h3>
                <div className="pm-feed">
                  {settlement.losers.length === 0 ? (
                    <span className="pm-muted">No settled traders.</span>
                  ) : (
                    settlement.losers.slice(0, 5).map((row) => (
                      <div key={`l-${row.agent_id}`} className="pm-feed-row">
                        Agent {row.agent_id} ({row.name}) · {row.yes_shares.toFixed(2)} shares · payout{" "}
                        {row.payout.toFixed(2)}
                      </div>
                    ))
                  )}
                </div>
              </section>
            </div>
          ) : (
            <p className="pm-muted small" style={{ marginBottom: 0 }}>
              This market is already resolved. Detailed winner/loser payout rows appear immediately after resolution.
            </p>
          )}
        </section>
      ) : (
        <section className="pm-panel pm-controls-panel">
          <h2 className="pm-controls-title">Trading controls</h2>
          <p className="pm-muted small" style={{ marginTop: 0 }}>
            <strong>1.</strong> Spawn adds traders to the global pool. <strong>2.</strong>{" "}
            <strong>Start trading</strong> turns on autonomous activity (nothing happens until then). With many agents,
            different IDs will show in the feed over time—not only one.
          </p>
          <div className="pm-controls-grid">
            <div className="pm-control-block">
              <span className="pm-control-label">Step 1 — Agents</span>
              <div className="pm-inline">
                <label>
                  Count
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={demoN}
                    onChange={(e) => setDemoN(Number(e.target.value))}
                    disabled={busy || resolveBusy}
                  />
                </label>
                <button type="button" className="pm-btn-secondary" disabled={busy || resolveBusy} onClick={spawnDemoAgents}>
                  Spawn agents
                </button>
              </div>
            </div>
            <div className="pm-control-block pm-control-primary">
              <span className="pm-control-label">Step 2 — Autonomous engine</span>
              <div className="pm-toolbar">
                <button type="button" className="pm-btn-primary" disabled={busy || resolveBusy || running} onClick={handleStart}>
                  Start trading
                </button>
                <button type="button" className="pm-btn-danger" disabled={busy || resolveBusy || !running} onClick={handleStop}>
                  Stop
                </button>
                <button type="button" className="pm-btn-secondary" disabled={busy || resolveBusy} onClick={() => setShowNews(true)}>
                  News event
                </button>
                <button
                  type="button"
                  className="pm-btn-success"
                  disabled={busy || resolveBusy}
                  onClick={() => setShowResolve(true)}
                >
                  Resolve market
                </button>
              </div>
              {running ? (
                <p className="pm-running-pill" aria-live="polite">
                  Running — autonomous traders are active for this market
                </p>
              ) : (
                <p className="pm-muted small" style={{ margin: "0.35rem 0 0" }}>
                  Not running yet — press <strong>Start trading</strong> after spawning agents.
                </p>
              )}
            </div>
          </div>
        </section>
      )}

      <div className="pm-two-col">
        <section className="pm-panel pm-grow">
          <h2>Live price</h2>
          <p className="pm-muted small" style={{ marginTop: "-0.25rem", marginBottom: "0.5rem" }}>
            <strong>Green</strong> = LMSR mid (inventory). <strong>Purple</strong> = mean belief over agents in this
            market.{" "}
            <strong>Amber dashed</strong> = scenario P* (ground truth). Same idea as Classic &quot;Mean belief vs
            price&quot;—mid can still pin near 0%/100% while beliefs sit near P*. Trade feed shows execution prices.
          </p>
          <div className="pm-chart-wrap">
            {chartData.length === 0 ? (
              <p className="pm-muted">Waiting for price samples…</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <XAxis dataKey="sec" tick={{ fontSize: 11 }} stroke="#64748b" unit="s" />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} stroke="#64748b" />
                  <Tooltip
                    formatter={(v: number | string, name: string) => [
                      typeof v === "number" ? `${v}%` : String(v),
                      name,
                    ]}
                  />
                  <Legend wrapperStyle={{ fontSize: "12px" }} />
                  {newsMarkers.map((m) => (
                    <ReferenceLine
                      key={`news-${m.id}`}
                      x={m.sec}
                      stroke="#dc2626"
                      strokeDasharray="3 3"
                      label={{ value: "News", position: "insideTopRight", fill: "#dc2626", fontSize: 10 }}
                    />
                  ))}
                  <Line
                    type="monotone"
                    dataKey="mid"
                    name="LMSR mid"
                    stroke="#22c55e"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="meanBelief"
                    name="Mean belief"
                    stroke="#7c3aed"
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="pStar"
                    name="P* (ground truth)"
                    stroke="#d97706"
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    dot={false}
                    connectNulls
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </section>

        {detail?.mechanism === "cda" && book ? (
          <section className="pm-panel">
            <h2>Order book</h2>
            <div className="pm-book">
              <div>
                <h3 className="pm-book-side asks">Asks</h3>
                <ul className="pm-book-list">
                  {(book.asks ?? []).slice(0, 8).map((a, i) => (
                    <li key={`a-${i}`}>
                      {(a.price * 100).toFixed(1)}¢ · {a.quantity.toFixed(2)}
                    </li>
                  ))}
                </ul>
              </div>
              <div>
                <h3 className="pm-book-side bids">Bids</h3>
                <ul className="pm-book-list">
                  {(book.bids ?? []).slice(0, 8).map((b, i) => (
                    <li key={`b-${i}`}>
                      {(b.price * 100).toFixed(1)}¢ · {b.quantity.toFixed(2)}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>
        ) : null}
      </div>

      <div className="pm-two-col">
        <section className="pm-panel">
          <h2>Trade feed</h2>
          <p className="pm-muted small" style={{ marginTop: "-0.35rem", marginBottom: "0.5rem" }}>
            Polls every 1s for new rows. If the <strong>server trade count</strong> below stops going up, the
            simulator isn&apos;t placing new fills (often: price pinned near 0%/100%, agents out of cash, or personality
            skips)—not a frozen UI.
          </p>
          <div className="pm-feed" aria-live="polite">
            {trades.length === 0 ? (
              <span className="pm-muted">No trades yet.</span>
            ) : (
              trades.map((t) => (
                <div key={`${t.trade_id}-${t.at}`} className="pm-feed-row">
                  <span className="mono">#{t.trade_id}</span>{" "}
                  <strong>{t.quantity >= 0 ? "BUY" : "SELL"}</strong> · Agent {t.agent_id} ·{" "}
                  {Math.abs(t.quantity).toFixed(2)} @ {(t.price * 100).toFixed(1)}¢
                </div>
              ))
            )}
          </div>
          {serverTradeTotal != null ? (
            <p className="pm-feed-meta" aria-live="polite">
              Server trade count: <strong>{serverTradeTotal}</strong>
              {trades.length > 0 ? (
                <>
                  {" "}
                  · newest on this page: <span className="mono">#{trades[trades.length - 1]?.trade_id}</span>
                </>
              ) : null}
            </p>
          ) : null}
        </section>

        <section className="pm-panel">
          <h2>Trader chat</h2>
          <p className="pm-muted small">Template or Ollama (see API env). New lines follow trades.</p>
          <div className="pm-comments" aria-live="polite">
            {comments.length === 0 ? (
              <span className="pm-muted">No comments yet — start trading to generate chatter.</span>
            ) : (
              comments.map((c) => (
                <div key={c.id} className="pm-comment">
                  <div className="pm-comment-meta">
                    Agent {c.agent_id} · {c.source === "llm" ? "LLM" : "template"}
                  </div>
                  <div>{c.text}</div>
                </div>
              ))
            )}
          </div>
        </section>
      </div>

      <section className="pm-panel">
        <h2>News history</h2>
        {newsHistory.length === 0 ? (
          <p className="pm-muted">No news events yet.</p>
        ) : (
          <div className="pm-feed" aria-live="polite">
            {newsHistory.map((ev) => (
              <div key={ev.id} className="pm-feed-row">
                <strong>{new Date(ev.at_timestamp).toLocaleString()}</strong> · {ev.headline} · affected{" "}
                {ev.n_affected} agents · mean belief {(ev.mean_belief_before * 100).toFixed(1)}% to{" "}
                {(ev.mean_belief_after * 100).toFixed(1)}%
              </div>
            ))}
          </div>
        )}
      </section>

      {showNews ? (
        <div className="pm-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="news-title">
          <div className="pm-modal">
            <h2 id="news-title">News event</h2>
            <p className="pm-muted">Shifts beliefs for the most signal-sensitive slice of agents (persistent).</p>
            <div className="pm-form">
              <label>
                Headline
                <input value={newsHeadline} onChange={(e) => setNewsHeadline(e.target.value)} />
              </label>
              <label>
                Target belief (0–1)
                <input
                  type="number"
                  step={0.01}
                  min={0.01}
                  max={0.99}
                  value={newsBelief}
                  onChange={(e) => setNewsBelief(Number(e.target.value))}
                />
              </label>
              <label>
                Affected fraction
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={newsFraction}
                  onChange={(e) => setNewsFraction(Number(e.target.value))}
                />
              </label>
              <label>
                Min signal sensitivity
                <input
                  type="number"
                  step={0.05}
                  min={0}
                  max={1}
                  value={newsMinSens}
                  onChange={(e) => setNewsMinSens(Number(e.target.value))}
                />
              </label>
              <div className="pm-modal-actions">
                <button type="button" className="pm-btn-ghost" onClick={() => setShowNews(false)} disabled={busy}>
                  Cancel
                </button>
                <button type="button" className="pm-btn-primary" disabled={busy} onClick={postNews}>
                  Publish news
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
      {showResolve ? (
        <div className="pm-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="resolve-title">
          <div className="pm-modal">
            <h2 id="resolve-title">Resolve market</h2>
            <p className="pm-muted">
              Outcome is drawn automatically from the market&apos;s ground truth probability (P*). This closes the
              market and settles payouts.
            </p>
            <div className="pm-form">
              <div className="pm-modal-actions">
                <button
                  type="button"
                  className="pm-btn-ghost"
                  onClick={() => setShowResolve(false)}
                  disabled={resolveBusy}
                >
                  Cancel
                </button>
                <button type="button" className="pm-btn-success" disabled={resolveBusy} onClick={resolveMarket}>
                  {resolveBusy ? "Resolving..." : "Confirm resolution"}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
