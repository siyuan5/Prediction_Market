import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

type AgentRow = {
  agent_id: number;
  name: string;
  cash: number;
  belief: number | null;
  avg_joined_belief?: number | null;
  rho: number | null;
};

type SortKey = "agent_id" | "name" | "avg_joined_belief" | "cash" | "rho";

export function AgentsPage() {
  const navigate = useNavigate();
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("agent_id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [deleteBusyId, setDeleteBusyId] = useState<number | null>(null);
  const [deleteAllBusy, setDeleteAllBusy] = useState(false);

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

  async function deleteAgent(agentId: number, name: string) {
    if (deleteAllBusy) return;
    if (!window.confirm(`Delete agent "${name}"? Trade history stays in the database.`)) return;
    setDeleteBusyId(agentId);
    setError(null);
    try {
      const res = await fetch(`/api/agents/${agentId}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await res.text());
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleteBusyId(null);
    }
  }

  async function deleteAllAgents() {
    if (agents.length === 0 || deleteAllBusy) return;
    if (
      !window.confirm(
        `Delete all ${agents.length} agents currently listed? ` +
          "Trade history stays in the database.",
      )
    ) {
      return;
    }
    setDeleteAllBusy(true);
    setError(null);
    try {
      const ids = agents.map((a) => a.agent_id);
      for (const agentId of ids) {
        const res = await fetch(`/api/agents/${agentId}`, { method: "DELETE" });
        if (!res.ok) throw new Error(await res.text());
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleteAllBusy(false);
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
      <p className="pm-sub">Global trader roster · average belief across all joined markets</p>

      {error ? (
        <p className="pm-error" role="alert">
          {error}
        </p>
      ) : null}

      <div className="pm-toolbar" style={{ justifyContent: "flex-end" }}>
        <button
          type="button"
          className="pm-btn-danger"
          disabled={deleteAllBusy || deleteBusyId != null || agents.length === 0}
          onClick={() => void deleteAllAgents()}
        >
          {deleteAllBusy ? "Deleting all..." : `Delete all agents (${agents.length})`}
        </button>
      </div>

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
                <button type="button" className="pm-th-btn" onClick={() => toggleSort("avg_joined_belief")}>
                  Avg joined belief {sortKey === "avg_joined_belief" ? (sortDir === "asc" ? "↑" : "↓") : ""}
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
              <th>Delete</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((a) => (
              <tr
                key={a.agent_id}
                className="pm-clickable-row"
                onClick={() => navigate(`/agents/${a.agent_id}`)}
              >
                <td>{a.agent_id}</td>
                <td>
                  <Link to={`/agents/${a.agent_id}`} onClick={(e) => e.stopPropagation()}>
                    {a.name}
                  </Link>
                </td>
                <td>{a.avg_joined_belief != null ? `${(a.avg_joined_belief * 100).toFixed(1)}%` : "—"}</td>
                <td>{a.rho != null ? a.rho.toFixed(2) : "—"}</td>
                <td>${a.cash?.toFixed(2) ?? "—"}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  <button
                    type="button"
                    className="pm-btn-danger"
                    disabled={deleteAllBusy || deleteBusyId === a.agent_id}
                    onClick={() => void deleteAgent(a.agent_id, a.name)}
                  >
                    {deleteBusyId === a.agent_id ? "Deleting..." : "Delete"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
