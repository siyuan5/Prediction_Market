import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

type AgentProfile = {
  agent_id: number;
  name: string;
  cash: number;
  belief: number | null;
  rho: number | null;
  personality: Record<string, unknown>;
  created_at?: string;
};

type AgentMarketRow = {
  market_id: number;
  title: string;
  status: string;
  mechanism: string;
  position: number;
  price: number;
  unrealized_pnl: number;
  trade_count: number;
  last_trade_at?: string | null;
};

type AgentTradeRow = {
  trade_id: string;
  market_id: number;
  market_title?: string | null;
  side?: string | null;
  quantity: number;
  price: number;
  price_before: number;
  cost: number;
  at?: string | null;
};

type AgentCommentRow = {
  id: number;
  market_id: number;
  market_title?: string | null;
  text: string;
  source?: string;
  at?: string;
  trade_id?: number;
};

type TradeSortKey = "at" | "market_title" | "quantity" | "price";

function fmtMoney(v: number | null | undefined) {
  return typeof v === "number" && Number.isFinite(v) ? `$${v.toFixed(2)}` : "--";
}

function fmtPct(v: number | null | undefined) {
  return typeof v === "number" && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "--";
}

function fmtSignedMoney(v: number) {
  const prefix = v > 0 ? "+" : "";
  return `${prefix}${fmtMoney(v)}`;
}

function fmtDate(v?: string | null) {
  if (!v) return "--";
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? v : d.toLocaleString();
}

function personalitySummary(personality: Record<string, unknown>) {
  const keys = ["participation_rate", "edge_threshold", "signal_sensitivity", "stubbornness", "trade_fraction"];
  const parts = keys
    .map((key) => {
      const raw = personality[key];
      const n = typeof raw === "number" ? raw : Number(raw);
      if (!Number.isFinite(n)) return null;
      return `${key.replaceAll("_", " ")} ${(n * 100).toFixed(0)}%`;
    })
    .filter((x): x is string => x != null);
  return parts.length > 0 ? parts.join(" · ") : "No personality profile recorded.";
}

export function AgentDetailPage() {
  const { agentId: agentParam } = useParams();
  const agentId = Number.parseInt(agentParam ?? "", 10);

  const [agent, setAgent] = useState<AgentProfile | null>(null);
  const [markets, setMarkets] = useState<AgentMarketRow[]>([]);
  const [trades, setTrades] = useState<AgentTradeRow[]>([]);
  const [comments, setComments] = useState<AgentCommentRow[]>([]);
  const [totalPnl, setTotalPnl] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<TradeSortKey>("at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const load = useCallback(async () => {
    if (!Number.isFinite(agentId)) return;
    try {
      const [agentRes, marketsRes, tradesRes, commentsRes] = await Promise.all([
        fetch(`/api/agents/${agentId}`),
        fetch(`/api/agents/${agentId}/markets`),
        fetch(`/api/agents/${agentId}/trades?limit=1000`),
        fetch(`/api/agents/${agentId}/comments`),
      ]);
      for (const res of [agentRes, marketsRes, tradesRes, commentsRes]) {
        if (!res.ok) throw new Error(await res.text());
      }
      const agentData = (await agentRes.json()) as AgentProfile;
      const marketData = (await marketsRes.json()) as { markets: AgentMarketRow[]; total_pnl?: number };
      const tradeData = (await tradesRes.json()) as { trades: AgentTradeRow[] };
      const commentData = (await commentsRes.json()) as { comments: AgentCommentRow[] };
      setAgent(agentData);
      setMarkets(marketData.markets ?? []);
      setTrades(tradeData.trades ?? []);
      setComments(commentData.comments ?? []);
      setTotalPnl(
        typeof marketData.total_pnl === "number"
          ? marketData.total_pnl
          : (marketData.markets ?? []).reduce((sum, row) => sum + row.unrealized_pnl, 0),
      );
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [agentId]);

  useEffect(() => {
    void load();
    const id = window.setInterval(load, 5000);
    return () => window.clearInterval(id);
  }, [load]);

  const sortedTrades = useMemo(() => {
    const rows = [...trades];
    const dir = sortDir === "asc" ? 1 : -1;
    rows.sort((a, b) => {
      if (sortKey === "at") {
        const av = a.at ? Date.parse(a.at) : 0;
        const bv = b.at ? Date.parse(b.at) : 0;
        return (av - bv) * dir;
      }
      const av = a[sortKey];
      const bv = b[sortKey];
      if (typeof av === "string" || typeof bv === "string") {
        return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
      }
      return ((av ?? 0) - (bv ?? 0)) * dir;
    });
    return rows;
  }, [trades, sortKey, sortDir]);

  function toggleSort(k: TradeSortKey) {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir(k === "at" ? "desc" : "asc");
    }
  }

  if (!Number.isFinite(agentId)) {
    return (
      <p className="pm-error">
        Invalid agent. <Link to="/agents">Back to agents</Link>
      </p>
    );
  }

  return (
    <div className="pm-page pm-detail">
      <nav className="pm-breadcrumb">
        <Link to="/">Markets</Link>
        <span className="pm-bc-sep">/</span>
        <Link to="/agents">Agents</Link>
        <span className="pm-bc-sep">/</span>
        <span>{agent?.name ?? `Agent #${agentId}`}</span>
      </nav>

      {error ? (
        <p className="pm-error" role="alert">
          {error}
        </p>
      ) : null}

      <header className="pm-detail-head">
        <div>
          <h1 className="pm-detail-title">{agent?.name ?? (loading ? "Loading..." : `Agent #${agentId}`)}</h1>
          <p className="pm-muted">
            Agent #{agentId} · belief {fmtPct(agent?.belief)} · rho {agent?.rho != null ? agent.rho.toFixed(2) : "--"}
          </p>
          <p className="pm-muted small">{agent ? personalitySummary(agent.personality ?? {}) : "Loading profile..."}</p>
        </div>
        <div className="pm-detail-yes-block">
          <div className="pm-yes-huge-label">Total PnL</div>
          <div className={totalPnl >= 0 ? "pm-yes-huge profit-pos" : "pm-yes-huge profit-neg"}>
            {fmtSignedMoney(totalPnl)}
          </div>
          <div className="pm-ts">Cash {fmtMoney(agent?.cash)}</div>
        </div>
      </header>

      <section className="pm-panel pm-table-wrap">
        <h2>Markets joined</h2>
        {markets.length === 0 ? (
          <p className="pm-muted">No market positions or trades yet.</p>
        ) : (
          <table className="pm-table">
            <thead>
              <tr>
                <th>Market</th>
                <th>Status</th>
                <th>Position</th>
                <th>Mark price</th>
                <th>Unrealized PnL</th>
                <th>Trades</th>
              </tr>
            </thead>
            <tbody>
              {markets.map((m) => (
                <tr key={m.market_id}>
                  <td>
                    <Link to={`/market/${m.market_id}`}>{m.title || `Market #${m.market_id}`}</Link>
                    <div className="pm-comment-meta">{m.mechanism.toUpperCase()}</div>
                  </td>
                  <td>{m.status}</td>
                  <td>{m.position.toFixed(2)}</td>
                  <td>{fmtPct(m.price)}</td>
                  <td className={m.unrealized_pnl >= 0 ? "profit-pos" : "profit-neg"}>
                    {fmtSignedMoney(m.unrealized_pnl)}
                  </td>
                  <td>{m.trade_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <div className="pm-two-col">
        <section className="pm-panel pm-table-wrap">
          <h2>Trade history</h2>
          {sortedTrades.length === 0 ? (
            <p className="pm-muted">No trades yet.</p>
          ) : (
            <table className="pm-table">
              <thead>
                <tr>
                  <th>
                    <button type="button" className="pm-th-btn" onClick={() => toggleSort("at")}>
                      Time {sortKey === "at" ? (sortDir === "asc" ? "up" : "down") : ""}
                    </button>
                  </th>
                  <th>
                    <button type="button" className="pm-th-btn" onClick={() => toggleSort("market_title")}>
                      Market {sortKey === "market_title" ? (sortDir === "asc" ? "up" : "down") : ""}
                    </button>
                  </th>
                  <th>Side</th>
                  <th>
                    <button type="button" className="pm-th-btn" onClick={() => toggleSort("quantity")}>
                      Qty {sortKey === "quantity" ? (sortDir === "asc" ? "up" : "down") : ""}
                    </button>
                  </th>
                  <th>
                    <button type="button" className="pm-th-btn" onClick={() => toggleSort("price")}>
                      Price {sortKey === "price" ? (sortDir === "asc" ? "up" : "down") : ""}
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedTrades.map((t) => (
                  <tr key={t.trade_id}>
                    <td>{fmtDate(t.at)}</td>
                    <td>
                      <Link to={`/market/${t.market_id}`}>{t.market_title || `Market #${t.market_id}`}</Link>
                    </td>
                    <td>{t.side ?? (t.quantity >= 0 ? "buy_yes" : "sell_yes")}</td>
                    <td>{t.quantity.toFixed(2)}</td>
                    <td>{fmtPct(t.price)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="pm-panel">
          <h2>Comments made</h2>
          <div className="pm-comments" aria-live="polite">
            {comments.length === 0 ? (
              <span className="pm-muted">No comments generated for this agent yet.</span>
            ) : (
              comments.map((c) => (
                <div key={`${c.market_id}-${c.id}`} className="pm-comment">
                  <div className="pm-comment-meta">
                    <Link to={`/market/${c.market_id}`}>{c.market_title || `Market #${c.market_id}`}</Link>
                    {" · "}
                    {c.source === "llm" ? "LLM" : "template"} · {fmtDate(c.at)}
                  </div>
                  <div>{c.text}</div>
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
