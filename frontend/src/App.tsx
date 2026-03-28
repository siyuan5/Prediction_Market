import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Mechanism = "lmsr" | "cda";
type BeliefMode = "gaussian" | "uniform" | "fixed" | "bimodal";

type SimulatePayload = {
  event_name: string;
  mechanism: Mechanism;
  seed: number;
  ground_truth: number;
  n_agents: number;
  n_rounds: number;
  initial_cash: number;
  b: number;
  initial_price: number;
  belief_mode: BeliefMode;
  belief_sigma: number;
  belief_fixed: number;
};

type SimComment = {
  round: number;
  agent_id: number;
  belief: number;
  text: string;
};

type SimMetrics = {
  total_rounds: number;
  price_series: number[];
  error_series: number[];
  trade_volume: number[];
  mean_belief_series: number[];
  mean_initial_belief: number;
  final_price: number | null;
  final_error: number | null;
  best_bid_series?: (number | null)[];
  best_ask_series?: (number | null)[];
};

type SettlementRow = {
  agent_id: number;
  belief: number;
  rho: number;
  cash_pre_settlement: number;
  yes_shares: number;
  terminal_wealth: number;
  profit: number;
};

type Settlement = {
  outcome: "yes" | "no";
  outcome_is_yes: boolean;
  p_star: number;
  initial_cash: number;
  payoff_per_yes_share: number;
  winners: SettlementRow[];
  losers: SettlementRow[];
};

type SimulateResponse = {
  event_name: string;
  metrics: SimMetrics;
  comments: SimComment[];
  comment_sampling?: { probability_per_agent_per_round: number };
  state: { ground_truth: number; mechanism: string; phase: number };
  settlement?: Settlement;
};

const defaultPayload: SimulatePayload = {
  event_name: "Will the launch ship by Friday?",
  mechanism: "lmsr",
  seed: 42,
  ground_truth: 0.7,
  n_agents: 50,
  n_rounds: 120,
  initial_cash: 100,
  b: 100,
  initial_price: 0.5,
  belief_mode: "gaussian",
  belief_sigma: 0.1,
  belief_fixed: 0.7,
};

const CHART_REVEAL_MS = 5000;

const tooltipStyle = {
  background: "#ffffff",
  border: "1px solid #e2e8f0",
  borderRadius: 12,
  boxShadow: "0 10px 28px rgba(15, 23, 42, 0.08)",
};

function fmtMoney(n: number) {
  const s = n.toFixed(2);
  return n >= 0 ? `+${s}` : s;
}

function buildChartRows(metrics: SimMetrics) {
  const n = metrics.price_series?.length ?? 0;
  const rows: {
    round: number;
    yes: number;
    no: number;
    meanBelief: number | null;
  }[] = [];
  for (let i = 0; i < n; i++) {
    const p = metrics.price_series[i];
    const mb = metrics.mean_belief_series[i];
    rows.push({
      round: i + 1,
      yes: Math.round(p * 1000) / 10,
      no: Math.round((1 - p) * 1000) / 10,
      meanBelief: mb != null ? Math.round(mb * 1000) / 10 : null,
    });
  }
  return rows;
}

export default function App() {
  const [form, setForm] = useState<SimulatePayload>(defaultPayload);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulateResponse | null>(null);

  const chartData = useMemo(
    () => (result ? buildChartRows(result.metrics) : []),
    [result],
  );

  /** 0 → 1 over CHART_REVEAL_MS so both charts draw left-to-right in sync */
  const [chartReveal01, setChartReveal01] = useState(0);

  useEffect(() => {
    if (chartData.length === 0) {
      setChartReveal01(0);
      return;
    }
    setChartReveal01(0);
    const start = performance.now();
    let raf = 0;
    let cancelled = false;
    const tick = (now: number) => {
      if (cancelled) return;
      const t = Math.min(1, (now - start) / CHART_REVEAL_MS);
      setChartReveal01(t);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [chartData]);

  const animatedChartData = useMemo(() => {
    const n = chartData.length;
    if (n === 0) return [];
    const count = Math.max(1, Math.ceil(chartReveal01 * n));
    return chartData.slice(0, count);
  }, [chartData, chartReveal01]);

  const roundDomainMax = Math.max(1, chartData.length);

  const commentsSorted = useMemo(() => {
    if (!result?.comments?.length) return [];
    return [...result.comments].sort((a, b) => {
      if (a.round !== b.round) return a.round - b.round;
      return a.agent_id - b.agent_id;
    });
  }, [result]);

  async function runSim() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as SimulateResponse;
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  function update<K extends keyof SimulatePayload>(key: K, value: SimulatePayload[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  const displayEvent = result?.event_name ?? form.event_name;

  return (
    <>
      <span className="event-badge">Phase 2 · signals every round</span>
      <h1>Prediction market playground</h1>
      <p className="sub">
        Pick LMSR or CDA, name your event, and set the true Yes chance. Agents see fresh signals each
        round and update beliefs. The main chart is implied Yes% (No is 100% minus that); comments are
        filler for now.
      </p>

      <div className="panel">
        <div className="form-grid">
          <label className="label-full">
            Event name
            <input
              type="text"
              placeholder="e.g. Will Team A win the match?"
              value={form.event_name}
              onChange={(e) => update("event_name", e.target.value)}
              maxLength={200}
            />
          </label>
          <label>
            Mechanism
            <select
              value={form.mechanism}
              onChange={(e) => update("mechanism", e.target.value as Mechanism)}
            >
              <option value="lmsr">LMSR (Team A)</option>
              <option value="cda">CDA (Team B)</option>
            </select>
          </label>
          <label>
            Ground truth (Yes prob.)
            <input
              type="number"
              step={0.01}
              min={0.01}
              max={0.99}
              value={form.ground_truth}
              onChange={(e) => update("ground_truth", Number(e.target.value))}
            />
          </label>
          <label>
            Agents
            <input
              type="number"
              min={2}
              max={500}
              value={form.n_agents}
              onChange={(e) => update("n_agents", Number(e.target.value))}
            />
          </label>
          <label>
            Rounds
            <input
              type="number"
              min={1}
              max={2000}
              value={form.n_rounds}
              onChange={(e) => update("n_rounds", Number(e.target.value))}
            />
          </label>
          <label>
            Seed
            <input
              type="number"
              value={form.seed}
              onChange={(e) => update("seed", Number(e.target.value))}
            />
          </label>
          {form.mechanism === "lmsr" ? (
            <label>
              LMSR liquidity (b)
              <input
                type="number"
                min={1}
                step={1}
                value={form.b}
                onChange={(e) => update("b", Number(e.target.value))}
              />
            </label>
          ) : null}
          {form.mechanism === "cda" ? (
            <label>
              CDA start price
              <input
                type="number"
                step={0.01}
                min={0.01}
                max={0.99}
                value={form.initial_price}
                onChange={(e) => update("initial_price", Number(e.target.value))}
              />
            </label>
          ) : null}
          <label>
            Belief init
            <select
              value={form.belief_mode}
              onChange={(e) => update("belief_mode", e.target.value as BeliefMode)}
            >
              <option value="gaussian">Gaussian around truth</option>
              <option value="uniform">Uniform random</option>
              <option value="fixed">Fixed (all same)</option>
              <option value="bimodal">Bimodal polarized</option>
            </select>
          </label>
          <label>
            Belief σ (gaussian)
            <input
              type="number"
              step={0.01}
              min={0}
              max={0.5}
              value={form.belief_sigma}
              onChange={(e) => update("belief_sigma", Number(e.target.value))}
              disabled={form.belief_mode !== "gaussian"}
            />
          </label>
          <label>
            Fixed belief
            <input
              type="number"
              step={0.01}
              min={0.01}
              max={0.99}
              value={form.belief_fixed}
              onChange={(e) => update("belief_fixed", Number(e.target.value))}
              disabled={form.belief_mode !== "fixed"}
            />
          </label>
          <button className="primary" type="button" disabled={loading} onClick={runSim}>
            {loading ? "Running…" : "Run simulation"}
          </button>
        </div>
        {error ? <p className="error">{error}</p> : null}
      </div>

      {result ? (
        <>
          <div className="panel" style={{ marginBottom: "1.1rem" }}>
            <strong style={{ fontSize: "1.15rem", display: "block", marginBottom: "0.35rem" }}>
              {displayEvent}
            </strong>
            <p className="sub" style={{ margin: 0 }}>
              Hidden true Yes rate: {(result.state.ground_truth * 100).toFixed(1)}% ·{" "}
              {result.state.mechanism.toUpperCase()} · {commentsSorted.length} sampled comments
            </p>
          </div>

          <div className="two-col">
            <div className="panel">
              <strong>Implied Yes % by round</strong>
              <p className="sub" style={{ marginBottom: "0.75rem" }}>
                How traders collectively price the chance the event happens, round by round. Hover a point
                for more detail.
              </p>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={animatedChartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      type="number"
                      dataKey="round"
                      domain={[1, roundDomainMax]}
                      tick={{ fill: "#64748b", fontSize: 11 }}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tickFormatter={(v) => `${v}%`}
                      tick={{ fill: "#64748b", fontSize: 11 }}
                    />
                    <Tooltip
                      content={({ active, payload, label }) => {
                        if (!active || !payload?.length) return null;
                        const yes = Number(payload[0].value);
                        const no = Math.round((100 - yes) * 10) / 10;
                        return (
                          <div
                            style={{
                              ...tooltipStyle,
                              padding: "10px 12px",
                              fontSize: "0.85rem",
                            }}
                          >
                            <div style={{ fontWeight: 600, marginBottom: 6 }}>Round {label}</div>
                            <div style={{ color: "#16a34a" }}>Yes: {yes}%</div>
                            <div style={{ color: "#64748b" }}>No: {no}% (implied)</div>
                          </div>
                        );
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="yes"
                      name="Implied Yes %"
                      stroke="#16a34a"
                      strokeWidth={2.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel">
              <strong>Agent comments</strong>
              <p className="sub" style={{ marginBottom: "0.5rem" }}>
                Filler from beliefs (LLM later). Each agent only rarely comments in a given round
                {result.comment_sampling != null
                  ? ` (~${(result.comment_sampling.probability_per_agent_per_round * 100).toFixed(1)}% chance)`
                  : ""}
                , but with many agents and rounds you still get a healthy sample.
              </p>
              <ul className="comments-list">
                {commentsSorted.map((c, i) => (
                  <li key={`${c.round}-${c.agent_id}-${i}`}>
                    <div className="meta">
                      Round {c.round} · Agent {c.agent_id} · {(c.belief * 100).toFixed(0)}% implied Yes
                      {c.belief > 0.5
                        ? " · leans Yes"
                        : c.belief < 0.5
                          ? " · leans No"
                          : " · even"}
                    </div>
                    {c.text}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="panel">
            <strong>Mean belief vs price</strong>
            <p className="sub" style={{ marginBottom: "0.75rem" }}>
              Crowd average belief compared to the traded Yes% (same scale).
            </p>
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={animatedChartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    type="number"
                    dataKey="round"
                    domain={[1, roundDomainMax]}
                    tick={{ fill: "#64748b", fontSize: 11 }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fill: "#64748b", fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    labelFormatter={(l) => `Round ${l}`}
                    formatter={(value: number, name: string) => [`${value}%`, name]}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="yes"
                    name="Price (Yes %)"
                    stroke="#0284c7"
                    strokeWidth={2}
                    fill="#0284c7"
                    fillOpacity={0.12}
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="meanBelief"
                    name="Mean belief"
                    stroke="#7c3aed"
                    strokeWidth={2}
                    fill="#7c3aed"
                    fillOpacity={0.1}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {result.settlement ? (
            <div className="panel">
              <strong>Settlement &amp; P&amp;L</strong>
              <p className="sub" style={{ marginBottom: "0.65rem" }}>
                After trading stops, the event resolves <strong>once</strong>: <strong>Yes</strong> with
                probability P* (your ground truth), else <strong>No</strong>. Each net <strong>Yes
                share</strong> pays $1 if Yes, $0 if No. Negative shares mean <strong>short Yes</strong>{" "}
                (you gain on <strong>No</strong> and lose on <strong>Yes</strong>)—so &quot;betting
                No&quot; shows up as negative Yes shares, not a separate vote tally.
              </p>
              <div className="settle-banner">
                <span
                  className={`settle-pill ${result.settlement.outcome_is_yes ? "yes" : "no"}`}
                >
                  Resolved {result.settlement.outcome_is_yes ? "YES" : "NO"}
                </span>
                <span className="sub" style={{ margin: 0 }}>
                  P(Yes) was {(result.settlement.p_star * 100).toFixed(1)}% · started with $
                  {result.settlement.initial_cash.toFixed(0)} each · payoff per Yes share: $
                  {result.settlement.payoff_per_yes_share.toFixed(0)}
                </span>
              </div>
              <p className="sub" style={{ marginBottom: "1rem" }}>
                <strong>Winners</strong> = highest profit after resolution; <strong>losers</strong> = lowest
                (even if still positive—those who did worst). You&apos;ll see real losers when the draw is
                No and agents are long Yes, or Yes with heavy short Yes.
              </p>
              <div className="leaderboard-grid">
                <div className="leaderboard-col">
                  <h3>Top 10 winners</h3>
                  <div className="leader-table-wrap">
                    <table className="leader-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Agent</th>
                          <th className="num">Yes shares</th>
                          <th className="num">Profit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.settlement.winners.map((r, i) => (
                          <tr key={r.agent_id}>
                            <td>{i + 1}</td>
                            <td>{r.agent_id}</td>
                            <td className="num">{r.yes_shares.toFixed(3)}</td>
                            <td
                              className={`num ${r.profit >= 0 ? "profit-pos" : "profit-neg"}`}
                            >
                              {fmtMoney(r.profit)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div className="leaderboard-col">
                  <h3>10 biggest losers</h3>
                  <div className="leader-table-wrap">
                    <table className="leader-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Agent</th>
                          <th className="num">Yes shares</th>
                          <th className="num">Profit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.settlement.losers.map((r, i) => (
                          <tr key={r.agent_id}>
                            <td>{i + 1}</td>
                            <td>{r.agent_id}</td>
                            <td className="num">{r.yes_shares.toFixed(3)}</td>
                            <td
                              className={`num ${r.profit < 0 ? "profit-neg" : "profit-pos"}`}
                            >
                              {fmtMoney(r.profit)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </>
      ) : null}
    </>
  );
}
