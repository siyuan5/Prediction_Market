import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

type MarketRow = {
  market_id: number;
  id?: number;
  title: string;
  mechanism: string;
  status: string;
  price: number;
  trade_count_24h?: number;
  active_agents_24h?: number;
  trade_count?: number;
  active_agents?: number;
};

export function MarketsHomePage() {
  const [markets, setMarkets] = useState<MarketRow[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [busy, setBusy] = useState(false);
  const [title, setTitle] = useState("Will ETH close above $4k by Friday?");
  const [mechanism, setMechanism] = useState<"lmsr" | "cda">("lmsr");
  const [groundTruth, setGroundTruth] = useState(0.65);
  const [b, setB] = useState(100);
  const [initialPrice, setInitialPrice] = useState(0.5);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/markets?status=all&limit=200&offset=0");
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as { markets: MarketRow[]; total: number };
      setMarkets(data.markets ?? []);
      setTotal(data.total ?? 0);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
    const id = window.setInterval(load, 2000);
    return () => window.clearInterval(id);
  }, [load]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const body: Record<string, unknown> = {
        mechanism,
        title: title.trim() || "Untitled market",
        ground_truth: groundTruth,
        b,
      };
      if (mechanism === "cda") body.initial_price = initialPrice;
      const res = await fetch("/api/market/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      setShowCreate(false);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="pm-page">
      <div className="pm-hero">
        <h1 className="pm-title">Markets</h1>
        <p className="pm-sub">Live prediction markets · {total} listed</p>
        <button type="button" className="pm-btn-primary" onClick={() => setShowCreate(true)}>
          Create market
        </button>
      </div>

      {error ? (
        <p className="pm-error" role="alert">
          {error}
        </p>
      ) : null}

      {showCreate ? (
        <div className="pm-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="create-title">
          <div className="pm-modal">
            <h2 id="create-title">New market</h2>
            <p className="pm-muted">Market only — add traders from the market page or Agents.</p>
            <form onSubmit={handleCreate} className="pm-form">
              <label>
                Question / title
                <input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={200} required />
              </label>
              <label>
                Mechanism
                <select value={mechanism} onChange={(e) => setMechanism(e.target.value as "lmsr" | "cda")}>
                  <option value="lmsr">LMSR (AMM)</option>
                  <option value="cda">CDA (order book)</option>
                </select>
              </label>
              <label>
                Implied true Yes probability (P*)
                <input
                  type="number"
                  step={0.01}
                  min={0.01}
                  max={0.99}
                  value={groundTruth}
                  onChange={(e) => setGroundTruth(Number(e.target.value))}
                />
              </label>
              {mechanism === "lmsr" ? (
                <label>
                  Liquidity (b)
                  <input type="number" min={1} step={1} value={b} onChange={(e) => setB(Number(e.target.value))} />
                </label>
              ) : (
                <label>
                  Initial CDA price
                  <input
                    type="number"
                    step={0.01}
                    min={0.01}
                    max={0.99}
                    value={initialPrice}
                    onChange={(e) => setInitialPrice(Number(e.target.value))}
                  />
                </label>
              )}
              <div className="pm-modal-actions">
                <button type="button" className="pm-btn-ghost" onClick={() => setShowCreate(false)} disabled={busy}>
                  Cancel
                </button>
                <button type="submit" className="pm-btn-primary" disabled={busy}>
                  {busy ? "Creating…" : "Create"}
                </button>
              </div>
            </form>
          </div>
        </div>
      ) : null}

      <div className="pm-card-grid">
        {markets.length === 0 ? (
          <p className="pm-muted">No markets yet. Create one to get started.</p>
        ) : (
          markets.map((m) => {
            const id = m.market_id ?? m.id ?? 0;
            const trades = m.trade_count_24h ?? m.trade_count ?? 0;
            const agents = m.active_agents_24h ?? m.active_agents ?? 0;
            const pct = Math.round((m.price ?? 0.5) * 1000) / 10;
            return (
              <Link key={id} to={`/market/${id}`} className="pm-market-card">
                <div className="pm-card-top">
                  <span className={`pm-status pm-status-${m.status}`}>{m.status}</span>
                  <span className="pm-mech">{m.mechanism?.toUpperCase()}</span>
                </div>
                <h3 className="pm-card-title">{m.title || `Market #${id}`}</h3>
                <div className="pm-card-yes">
                  <span className="pm-yes-label">Yes</span>
                  <span className="pm-yes-pct">{pct}%</span>
                </div>
                <div className="pm-card-meta">
                  <span>{trades} trades</span>
                  <span>{agents} active traders</span>
                </div>
              </Link>
            );
          })
        )}
      </div>
    </div>
  );
}
