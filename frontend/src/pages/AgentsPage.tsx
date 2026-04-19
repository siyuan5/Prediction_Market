import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

type AgentRow = {
  agent_id: number;
  name: string;
  cash: number;
  belief: number | null;
  rho: number | null;
};

type SortKey = "agent_id" | "name" | "belief" | "cash" | "rho";

export function AgentsPage() {
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("agent_id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [edits, setEdits] = useState<Record<number, string>>({});
  const [busyId, setBusyId] = useState<number | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch("/api/agents?limit=500&offset=0");
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as { agents: AgentRow[] };
      setAgents(data.agents ?? []);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
    const id = window.setInterval(load, 5000);
    return () => window.clearInterval(id);
  }, [load]);

  const sorted = useMemo(() => {
    const rows = [...agents];
    const dir = sortDir === "asc" ? 1 : -1;
    rows.sort((a, b) => {
      const va = a[sortKey];
      const vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "string" && typeof vb === "string") return va.localeCompare(vb) * dir;
      return ((va as number) - (vb as number)) * dir;
    });
    return rows;
  }, [agents, sortKey, sortDir]);

  function toggleSort(k: SortKey) {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir("asc");
    }
  }

  async function applyBelief(agentId: number) {
    const raw = edits[agentId]?.trim();
    if (raw === undefined || raw === "") return;
    const v = Number.parseFloat(raw);
    if (!Number.isFinite(v) || v < 0.01 || v > 0.99) {
      setError("Belief must be between 0.01 and 0.99");
      return;
    }
    setBusyId(agentId);
    setError(null);
    try {
      const res = await fetch(`/api/agents/${agentId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ belief: v }),
      });
      if (!res.ok) throw new Error(await res.text());
      setEdits((e) => ({ ...e, [agentId]: "" }));
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="pm-page">
      <nav className="pm-breadcrumb">
        <Link to="/">Markets</Link>
        <span className="pm-bc-sep">/</span>
        <span>Agents</span>
      </nav>
      <h1 className="pm-title">Agents</h1>
      <p className="pm-sub">Global trader roster · PATCH belief for manual shocks</p>

      {error ? (
        <p className="pm-error" role="alert">
          {error}
        </p>
      ) : null}

      <div className="pm-panel pm-table-wrap">
        <table className="pm-table">
          <thead>
            <tr>
              <th>
                <button type="button" className="pm-th-btn" onClick={() => toggleSort("agent_id")}>
                  ID {sortKey === "agent_id" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                </button>
              </th>
              <th>
                <button type="button" className="pm-th-btn" onClick={() => toggleSort("name")}>
                  Name {sortKey === "name" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                </button>
              </th>
              <th>
                <button type="button" className="pm-th-btn" onClick={() => toggleSort("belief")}>
                  Belief {sortKey === "belief" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                </button>
              </th>
              <th>
                <button type="button" className="pm-th-btn" onClick={() => toggleSort("rho")}>
                  ρ {sortKey === "rho" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                </button>
              </th>
              <th>
                <button type="button" className="pm-th-btn" onClick={() => toggleSort("cash")}>
                  Cash {sortKey === "cash" ? (sortDir === "asc" ? "↑" : "↓") : ""}
                </button>
              </th>
              <th>Set belief</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((a) => (
              <tr key={a.agent_id}>
                <td>{a.agent_id}</td>
                <td>{a.name}</td>
                <td>{a.belief != null ? `${(a.belief * 100).toFixed(1)}%` : "—"}</td>
                <td>{a.rho != null ? a.rho.toFixed(2) : "—"}</td>
                <td>${a.cash?.toFixed(2) ?? "—"}</td>
                <td>
                  <div className="pm-inline tight">
                    <input
                      className="pm-input-sm"
                      placeholder="0–1"
                      value={edits[a.agent_id] ?? ""}
                      onChange={(e) => setEdits((x) => ({ ...x, [a.agent_id]: e.target.value }))}
                    />
                    <button
                      type="button"
                      className="pm-btn-secondary"
                      disabled={busyId === a.agent_id}
                      onClick={() => applyBelief(a.agent_id)}
                    >
                      Apply
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
