/**
 * Prediction market UI: configure a run, then either
 *   (1) one-shot POST /api/simulate or streaming /api/simulate/stream, or
 *   (2) session mode — start → step/pause → optional belief shift → finish + settlement.
 * Charts consume `metrics` series; comments panel shows template or LLM lines from the API.
 */
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

type TradeFlow = "buy_yes" | "sell_yes" | "hold";

type SimComment = {
  round: number;
  agent_id: number;
  belief: number;
  text: string;
  source?: "llm" | "template";
  trade_flow?: TradeFlow;
};

type BeliefShiftEvent = {
  round: number;
  n_agents_shifted: number;
  agent_ids: number[];
  new_belief: number | null;
  delta: number | null;
  before_mean: number;
  after_mean: number;
};

type SimState = {
  round: number;
  price: number;
  error: number;
  mean_belief: number;
  ground_truth: number;
  mechanism: string;
  phase: number;
  belief_shift_events?: BeliefShiftEvent[];
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

type CommentSampling = {
  probability_per_agent_per_round: number;
  llm_budget_initial?: number;
  llm_budget_remaining?: number;
  max_comments_per_event?: number | null;
  comments_returned?: number;
  comments_so_far?: number;
};

type SimulateResponse = {
  event_name: string;
  metrics: SimMetrics;
  comments: SimComment[];
  comment_sampling?: CommentSampling;
  state: SimState;
  settlement?: Settlement;
};

type SessionSnapshot = {
  session_id: string;
  target_rounds: number;
  round: number;
  done: boolean;
  state: SimState;
  metrics: SimMetrics;
  comments: SimComment[];
  comment_sampling?: CommentSampling;
  rounds_advanced?: number;
  shift_event?: BeliefShiftEvent;
};

type StreamTick = {
  type: "tick";
  state: SimState;
  mean_initial_belief: number;
  append: {
    price: number;
    mean_belief: number;
    error: number;
    trade_volume: number;
  };
  new_comments: SimComment[];
  comment_sampling: CommentSampling;
  append_best_bid?: number | null;
  append_best_ask?: number | null;
};

function mergeStreamTick(
  prev: SimulateResponse | null,
  tick: StreamTick,
  form: SimulatePayload,
): SimulateResponse {
  const { append } = tick;
  const price_series = [...(prev?.metrics.price_series ?? []), append.price];
  const mean_belief_series = [...(prev?.metrics.mean_belief_series ?? []), append.mean_belief];
  const error_series = [...(prev?.metrics.error_series ?? []), append.error];
  const trade_volume = [...(prev?.metrics.trade_volume ?? []), append.trade_volume];
  let best_bid_series = prev?.metrics.best_bid_series;
  let best_ask_series = prev?.metrics.best_ask_series;
  if (tick.append_best_bid !== undefined) {
    best_bid_series = [...(best_bid_series ?? []), tick.append_best_bid];
  }
  if (tick.append_best_ask !== undefined) {
    best_ask_series = [...(best_ask_series ?? []), tick.append_best_ask];
  }
  const metrics: SimMetrics = {
    total_rounds: tick.state.round,
    price_series,
    mean_belief_series,
    error_series,
    trade_volume,
    mean_initial_belief: tick.mean_initial_belief,
    final_price: append.price,
    final_error: append.error,
  };
  if (best_bid_series != null) metrics.best_bid_series = best_bid_series;
  if (best_ask_series != null) metrics.best_ask_series = best_ask_series;
  return {
    event_name: form.event_name,
    metrics,
    state: tick.state,
    comments: [...(prev?.comments ?? []), ...tick.new_comments],
    comment_sampling: tick.comment_sampling,
  };
}

function ResultsSkeleton({
  eventName,
  mechanism,
  groundTruth,
}: {
  eventName: string;
  mechanism: string;
  groundTruth: number;
}) {
  return (
    <>
      <div className="panel skeleton-edge" style={{ marginBottom: "1.1rem" }}>
        <div className="skeleton-title" />
        <div className="skeleton-line skeleton-line-md" style={{ marginTop: "0.5rem" }} />
        <p className="sub" style={{ margin: "0.65rem 0 0", color: "var(--accent)", fontWeight: 600 }}>
          Running simulation…
        </p>
      </div>
      <div className="two-col">
        <div className="panel skeleton-edge">
          <div className="skeleton-h2" />
          <div className="skeleton-line skeleton-line-long" />
          <div className="chart-wrap chart-skeleton-wrap">
            <div className="chart-skeleton" />
            <p className="skeleton-status">Charts stream in round by round</p>
          </div>
        </div>
        <div className="panel skeleton-edge">
          <div className="skeleton-h2" />
          <div className="skeleton-line skeleton-line-long" />
          <div style={{ marginTop: "0.75rem" }}>
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="skeleton-comment-row">
                <div className="skeleton-comment-meta" />
                <div className={`skeleton-comment-body ${i % 2 ? "short" : ""}`} />
              </div>
            ))}
          </div>
          <p className="sub" style={{ margin: "0.65rem 0 0", fontSize: "0.82rem" }}>
            Comments appear as agents &quot;speak&quot;
          </p>
        </div>
      </div>
      <div className="panel skeleton-edge">
        <div className="skeleton-h2" style={{ width: "42%" }} />
        <div className="skeleton-line skeleton-line-long" />
        <div className="chart-wrap chart-skeleton-wrap">
          <div className="chart-skeleton" />
        </div>
      </div>
      <p className="sub" style={{ margin: "-0.35rem 0 0", fontSize: "0.82rem", color: "var(--muted)" }}>
        {eventName} · {(groundTruth * 100).toFixed(0)}% truth · {mechanism.toUpperCase()}
      </p>
    </>
  );
}

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
const COMMENT_PROB = 0.01;

function tradeFlowDescription(flow: TradeFlow | undefined): string {
  if (flow === "buy_yes") return " · bought Yes";
  if (flow === "sell_yes") return " · sold Yes";
  if (flow === "hold") return " · no trade";
  return "";
}

function snapshotToResult(snapshot: SessionSnapshot, eventName: string): SimulateResponse {
  return {
    event_name: eventName,
    metrics: snapshot.metrics,
    comments: snapshot.comments,
    state: snapshot.state,
    comment_sampling:
      snapshot.comment_sampling ?? { probability_per_agent_per_round: COMMENT_PROB },
  };
}

function parseAgentIds(raw: string): number[] | undefined {
  const t = raw.trim();
  if (!t) return undefined;
  const ids = t
    .split(/[\s,]+/)
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && Number.isInteger(n) && n >= 0);
  return ids.length ? ids : undefined;
}

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

export function ClassicDashboard() {
  const [form, setForm] = useState<SimulatePayload>(defaultPayload);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulateResponse | null>(null);
  const [interactiveMode, setInteractiveMode] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionBusy, setSessionBusy] = useState(false);
  const [roundsPerStep, setRoundsPerStep] = useState(1);
  const [shockMode, setShockMode] = useState<"new_belief" | "delta">("new_belief");
  const [shockValue, setShockValue] = useState(0.65);
  const [shockAgentIds, setShockAgentIds] = useState("");
  const [shiftNotice, setShiftNotice] = useState<string | null>(null);
  const [targetRounds, setTargetRounds] = useState(0);
  /** After a streamed run, keep charts instant (skip end-of-run reveal animation). */
  const [streamChartLock, setStreamChartLock] = useState(false);

  const busy = loading || sessionBusy;

  const chartData = useMemo(
    () => (result ? buildChartRows(result.metrics) : []),
    [result],
  );

  const chartLive =
    interactiveMode ||
    streamChartLock ||
    (Boolean(loading) && Boolean(result));

  /** 0 → 1 over CHART_REVEAL_MS so both charts draw left-to-right in sync (skipped in interactive mode) */
  const [chartReveal01, setChartReveal01] = useState(0);

  useEffect(() => {
    if (chartData.length === 0) {
      setChartReveal01(0);
      return;
    }
    if (chartLive) {
      setChartReveal01(1);
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
  }, [chartData, chartLive]);

  const animatedChartData = useMemo(() => {
    const n = chartData.length;
    if (n === 0) return [];
    if (chartLive) return chartData;
    const count = Math.max(1, Math.ceil(chartReveal01 * n));
    return chartData.slice(0, count);
  }, [chartData, chartReveal01, chartLive]);

  const roundDomainMax = Math.max(1, chartData.length);

  const commentsSorted = useMemo(() => {
    if (!result?.comments?.length) return [];
    return [...result.comments].sort((a, b) => {
      if (a.round !== b.round) return a.round - b.round;
      return a.agent_id - b.agent_id;
    });
  }, [result]);

  useEffect(() => {
    if (interactiveMode) return;
    if (!sessionId) return;
    const id = sessionId;
    void fetch(`/api/session/${encodeURIComponent(id)}`, { method: "DELETE" });
    setSessionId(null);
    setResult(null);
    setShiftNotice(null);
  }, [interactiveMode, sessionId]);

  async function runSim() {
    setLoading(true);
    setError(null);
    setResult(null);
    setStreamChartLock(false);
    const body = JSON.stringify(form);

    async function fetchSimulateJson(): Promise<SimulateResponse> {
      const res = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      return (await res.json()) as SimulateResponse;
    }

    try {
      const res = await fetch("/api/simulate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      if (!res.body) {
        const data = await fetchSimulateJson();
        setResult(data);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          const msg = JSON.parse(trimmed) as StreamTick | (SimulateResponse & { type: string });
          if (msg.type === "tick") {
            setResult((prev) => mergeStreamTick(prev, msg as StreamTick, form));
          } else if (msg.type === "done") {
            const { type: _t, ...rest } = msg as SimulateResponse & { type: "done" };
            setResult(rest);
            setStreamChartLock(true);
          }
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      try {
        const data = await fetchSimulateJson();
        setResult(data);
        setError(null);
      } catch {
        setResult(null);
      }
    } finally {
      setLoading(false);
    }
  }

  async function startInteractiveSession() {
    setSessionBusy(true);
    setError(null);
    setShiftNotice(null);
    try {
      const res = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const snap = (await res.json()) as SessionSnapshot;
      setSessionId(snap.session_id);
      setTargetRounds(snap.target_rounds);
      setResult(snapshotToResult(snap, form.event_name));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSessionId(null);
    } finally {
      setSessionBusy(false);
    }
  }

  async function sessionStep() {
    if (!sessionId) return;
    setSessionBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/session/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, rounds: roundsPerStep }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const snap = (await res.json()) as SessionSnapshot;
      setResult(snapshotToResult(snap, form.event_name));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSessionBusy(false);
    }
  }

  async function sessionShift() {
    if (!sessionId) return;
    setSessionBusy(true);
    setError(null);
    try {
      const payload: Record<string, unknown> = {
        session_id: sessionId,
        agent_ids: parseAgentIds(shockAgentIds),
      };
      if (shockMode === "new_belief") {
        const v = shockValue;
        if (v < 0.01 || v > 0.99) {
          throw new Error("Belief must be between 0.01 and 0.99");
        }
        payload.new_belief = v;
      } else {
        payload.delta = shockValue;
      }
      const res = await fetch("/api/session/shift", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const snap = (await res.json()) as SessionSnapshot;
      setResult(snapshotToResult(snap, form.event_name));
      if (snap.shift_event) {
        const ev = snap.shift_event;
        const kind = ev.new_belief != null ? `set to ${(ev.new_belief * 100).toFixed(0)}%` : `Δ ${ev.delta != null ? (ev.delta * 100).toFixed(1) : ""}%`;
        setShiftNotice(`After round ${ev.round}: shifted ${ev.n_agents_shifted} agents (${kind}). Mean belief ${(ev.before_mean * 100).toFixed(1)}% → ${(ev.after_mean * 100).toFixed(1)}%.`);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSessionBusy(false);
    }
  }

  async function sessionFinish() {
    if (!sessionId) return;
    setSessionBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/session/finish", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as SimulateResponse;
      setSessionId(null);
      setShiftNotice(null);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSessionBusy(false);
    }
  }

  async function sessionCancel() {
    if (!sessionId) return;
    setSessionBusy(true);
    setError(null);
    try {
      await fetch(`/api/session/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      setSessionId(null);
      setShiftNotice(null);
      setResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSessionBusy(false);
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
        round and update beliefs. Use <strong>Interactive run</strong> to advance one or more rounds at
        a time, inject belief shocks between steps, then finish for settlement. The main chart is
        implied Yes% (No is 100% minus that); comments are filler for now.
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
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={interactiveMode}
              onChange={(e) => setInteractiveMode(e.target.checked)}
              disabled={busy}
            />
            Interactive run (pause, shift beliefs, then finish)
          </label>
          {!interactiveMode ? (
            <button className="primary" type="button" disabled={busy} onClick={runSim}>
              {loading ? "Running…" : "Run simulation"}
            </button>
          ) : !sessionId ? (
            <button className="primary" type="button" disabled={busy} onClick={startInteractiveSession}>
              {sessionBusy ? "Starting…" : "Start interactive session"}
            </button>
          ) : null}
        </div>

        {interactiveMode && sessionId ? (
          <div className="interactive-dock">
            <div className="interactive-dock-header">
              <span className="interactive-dock-badge" aria-live="polite">
                Live session
              </span>
              <span className="interactive-dock-round">
                Round <strong>{result?.state.round ?? 0}</strong>
                <span className="interactive-dock-round-total"> / {targetRounds}</span>
                {result && result.state.round >= targetRounds ? (
                  <span className="interactive-done-pill">Target reached</span>
                ) : null}
              </span>
            </div>
            <div
              className="interactive-progress-track"
              role="progressbar"
              aria-valuenow={result?.state.round ?? 0}
              aria-valuemin={0}
              aria-valuemax={targetRounds || 1}
            >
              <div
                className="interactive-progress-fill"
                style={{
                  width: `${targetRounds > 0 ? Math.min(100, ((result?.state.round ?? 0) / targetRounds) * 100) : 0}%`,
                }}
              />
            </div>
            <div className="interactive-toolbar">
              <div className="interactive-toolbar-group">
                <label className="interactive-compact-label">
                  Rounds / step
                  <input
                    className="interactive-step-input"
                    type="number"
                    min={1}
                    max={100}
                    value={roundsPerStep}
                    onChange={(e) =>
                      setRoundsPerStep(Math.max(1, Math.min(100, Number(e.target.value) || 1)))
                    }
                    disabled={busy}
                  />
                </label>
                <button className="secondary" type="button" disabled={busy} onClick={sessionStep}>
                  {sessionBusy ? "…" : "Step"}
                </button>
              </div>
              <div className="interactive-toolbar-group">
                <button className="primary" type="button" disabled={busy} onClick={sessionFinish}>
                  Finish &amp; settle
                </button>
                <button className="danger-outline" type="button" disabled={busy} onClick={sessionCancel}>
                  Cancel
                </button>
              </div>
            </div>
            <div className="interactive-shock-block">
              <span className="interactive-shock-title">Belief shock</span>
              <p className="interactive-shock-hint">
                Apply between steps. Leave agent IDs empty to affect everyone, or list IDs separated by
                commas.
              </p>
              <div className="interactive-shock-grid">
                <label>
                  Mode
                  <select
                    value={shockMode}
                    onChange={(e) => setShockMode(e.target.value as "new_belief" | "delta")}
                    disabled={busy}
                  >
                    <option value="new_belief">Set belief to…</option>
                    <option value="delta">Add Δ to belief</option>
                  </select>
                </label>
                <label>
                  {shockMode === "new_belief" ? "New belief (0–1)" : "Delta (e.g. +0.05)"}
                  <input
                    type="number"
                    step={0.01}
                    value={shockValue}
                    onChange={(e) => setShockValue(Number(e.target.value))}
                    disabled={busy}
                  />
                </label>
                <label className="interactive-shock-span">
                  Agent IDs (optional)
                  <input
                    type="text"
                    placeholder="e.g. 0, 3, 12"
                    value={shockAgentIds}
                    onChange={(e) => setShockAgentIds(e.target.value)}
                    disabled={busy}
                  />
                </label>
                <button className="secondary" type="button" disabled={busy} onClick={sessionShift}>
                  Apply shock
                </button>
              </div>
              {shiftNotice ? <p className="interactive-shift-notice">{shiftNotice}</p> : null}
              {result?.state.belief_shift_events && result.state.belief_shift_events.length > 0 ? (
                <ul className="shift-log">
                  {result.state.belief_shift_events.map((ev, i) => (
                    <li key={i}>
                      Round {ev.round}: {ev.n_agents_shifted} agents — mean{" "}
                      {(ev.before_mean * 100).toFixed(1)}% → {(ev.after_mean * 100).toFixed(1)}%
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </div>
        ) : null}
        {error ? <p className="error">{error}</p> : null}
      </div>

      {result || (loading && !interactiveMode) ? (
        result ? (
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
                Short lines from random traders about your event
                {typeof result.comment_sampling?.max_comments_per_event === "number"
                  ? ` (max ${result.comment_sampling.max_comments_per_event})`
                  : result.comment_sampling?.max_comments_per_event === null
                    ? " (no cap)"
                    : ""}
                . <strong>LLM</strong> = written by a local model via{" "}
                <a href="https://ollama.com" target="_blank" rel="noreferrer">
                  Ollama
                </a>
                ; <strong>template</strong> = preset text if Ollama isn’t running or slots are used up. Each
                line notes net <strong>bought Yes</strong>, <strong>sold Yes</strong>, or <strong>no trade</strong>{" "}
                that round.
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
                      {tradeFlowDescription(c.trade_flow)}
                      {c.source === "llm" ? " · LLM" : c.source === "template" ? " · template" : ""}
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
        ) : (
          <ResultsSkeleton
            eventName={form.event_name}
            mechanism={form.mechanism}
            groundTruth={form.ground_truth}
          />
        )
      ) : null}
    </>
  );
}
