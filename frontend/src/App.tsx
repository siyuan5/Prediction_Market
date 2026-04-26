import { NavLink, Route, Routes } from "react-router-dom";
import { ClassicDashboard } from "./ClassicDashboard";
import { AgentDetailPage } from "./pages/AgentDetailPage";
import { AgentsPage } from "./pages/AgentsPage";
import { MarketDetailPage } from "./pages/MarketDetailPage";
import { MarketsHomePage } from "./pages/MarketsHomePage";

function Layout() {
  return (
    <div className="pm-root">
      <header className="pm-topnav">
        <NavLink to="/" className={({ isActive }) => (isActive ? "pm-nav active" : "pm-nav")} end>
          Markets
        </NavLink>
        <NavLink to="/agents" className={({ isActive }) => (isActive ? "pm-nav active" : "pm-nav")}>
          Agents
        </NavLink>
        <NavLink to="/classic" className={({ isActive }) => (isActive ? "pm-nav active" : "pm-nav")}>
          Classic sim
        </NavLink>
      </header>
      <main className="pm-main">
        <Routes>
          <Route path="/" element={<MarketsHomePage />} />
          <Route path="/market/:marketId" element={<MarketDetailPage />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/agents/:agentId" element={<AgentDetailPage />} />
          <Route path="/classic" element={<ClassicDashboard />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return <Layout />;
}
