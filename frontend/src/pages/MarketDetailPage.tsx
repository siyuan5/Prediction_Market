import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type Detail = {
  market_id: number;
  title: string;
  mechanism: string;
  status: string;
  ground_truth: number;
  price: number;
  trade_count: number;
  active_agents: number;
};

type PriceSnap = {
  market_id: number;
  price: number;
  best_bid: number | null;
  best_ask: number | null;
  last_trade_price?: number | null;
  timestamp: string;
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

export function MarketDetailPage() {
  const { marketId: midParam } = useParams();
  const marketId = Number.parseInt(midParam ?? "", 10);

  const [detail, setDetail] = useState<Detail | null>(null);
  const [price, setPrice] = useState<PriceSnap | null>(null);
  const [trades, setTrades] = useState<TradeRow[]>([]);
  const [book, setBook] = useState<{ bids: { price: number; quantity: number }[]; asks: { price: number; quantity: number }[] } | null>(null);
  const [comments, setComments] = useState<CommentRow[]>([]);
  const [chartRows, setChartRows] = useState<{ t: number; p: number }[]>([]);
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

  const loadDetail = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    try {
      const res = await fetch(`/api/market/${marketId}/detail`);
      if (!res.ok) throw new Error(await res.text());
      setDetail((await res.json()) as Detail);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [marketId]);

  const fetchPrice = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    const res = await fetch(`/api/market/${marketId}/price`);
    if (!res.ok) throw new Error(await res.text());
    const p = (await res.json()) as PriceSnap;
    setPrice(p);
    const t = Date.now();
    setChartRows((rows) => {
      const next = [...rows, { t, p: p.price }];
      return next.slice(-240);
    });
  }, [marketId]);

  const fetchTrades = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    const since = lastTradeIdRef.current;
    const q = since > 0 ? `?since=${since}&limit=80` : "?limit=80";
    const res = await fetch(`/api/market/${marketId}/trades${q}`);
    if (!res.ok) throw new Error(await res.text());
    const data = (await res.json()) as { trades: TradeRow[] };
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

  const tickComments = useCallback(async () => {
    if (!Number.isFinite(marketId)) return;
    await fetch(`/api/market/${marketId}/comments/tick`, { method: "POST" });
    await fetchComments();
  }, [marketId, fetchComments]);

  useEffect(() => {
    void loadDetail();
    const id = window.setInterval(loadDetail, 3000);
    return () => window.clearInterval(id);
  }, [loadDetail]);

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

  async function handleStart() {
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
    if (!Number.isFinite(marketId) || !detail) return;
    setBusy(true);
    setError(null);
    const gt = detail.ground_truth;
    try {
      const tag = Date.now();
      for (let i = 0; i < demoN; i += 1) {
        const beliefNoise = (Math.random() - 0.5) * 0.2;
        const belief = Math.min(0.99, Math.max(0.01, gt + beliefNoise));
        const res = await fetch("/api/agents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: `trader-${tag}-${i}`,
            cash: 100.0,
            belief,
            rho: 1.0,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
      }
      await loadDetail();
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
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const chartData = useMemo(() => {
    if (chartRows.length === 0) return [];
    const t0 = chartRows[0].t;
    return chartRows.map((r, i) => ({
      i,
      sec: Math.round((r.t - t0) / 100) / 10,
      yes: Math.round(r.p * 1000) / 10,
    }));
  }, [chartRows]);

  if (!Number.isFinite(marketId)) {
    return (
      <p className="pm-error">
        Invalid market. <Link to="/">Back</Link>
      </p>
    );
  }

  const yesPct = price ? Math.round(price.price * 1000) / 10 : null;

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
            trades · {detail?.active_agents ?? 0} traders
          </p>
        </div>
        <div className="pm-detail-yes-block" aria-live="polite">
          <div className="pm-yes-huge-label">Chance · Yes</div>
          <div className="pm-yes-huge">{yesPct != null ? `${yesPct}%` : "—"}</div>
          {price?.timestamp ? <div className="pm-ts">{price.timestamp}</div> : null}
        </div>
      </header>

      <div className="pm-toolbar">
        <button type="button" className="pm-btn-primary" disabled={busy || running} onClick={handleStart}>
          Start trading
        </button>
        <button type="button" className="pm-btn-danger" disabled={busy || !running} onClick={handleStop}>
          Stop
        </button>
        <button type="button" className="pm-btn-secondary" disabled={busy} onClick={() => setShowNews(true)}>
          News event
        </button>
      </div>

      <section className="pm-panel">
        <h2>Prepare demo traders</h2>
        <p className="pm-muted">
          Creates global agents (not tied to this market) with beliefs noisy around P*. They discover markets via the
          API when trading is running.
        </p>
        <div className="pm-inline">
          <label>
            Count
            <input
              type="number"
              min={1}
              max={200}
              value={demoN}
              onChange={(e) => setDemoN(Number(e.target.value))}
              disabled={busy}
            />
          </label>
          <button type="button" className="pm-btn-secondary" disabled={busy} onClick={spawnDemoAgents}>
            Spawn agents
          </button>
        </div>
      </section>

      <div className="pm-two-col">
        <section className="pm-panel pm-grow">
          <h2>Live price</h2>
          <div className="pm-chart-wrap">
            {chartData.length === 0 ? (
              <p className="pm-muted">Waiting for price samples…</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <XAxis dataKey="sec" tick={{ fontSize: 11 }} stroke="#64748b" unit="s" />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11 }} stroke="#64748b" />
                  <Tooltip formatter={(v: number) => [`${v}%`, "Yes"]} />
                  <Line type="monotone" dataKey="yes" stroke="#22c55e" strokeWidth={2} dot={false} isAnimationActive={false} />
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
    </div>
  );
}
