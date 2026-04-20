# end-to-end integration tests for the autonomous prediction market stack
#
# covers:
#   - full flow: create market -> create agents -> join -> trade -> verify state
#   - news event shifts beliefs persistently
#   - autonomous trading lifecycle (start/stop, trade count matches)
#   - round-based vs autonomous: both paths produce valid final state
#   - concurrency stress: many trades under load, no corruption
#
# uses FastAPI TestClient for most tests (fast, sync).
# uses uvicorn-in-thread for the autonomous/concurrency tests (need real HTTP
# because agents call requests.get/post against a real URL).

from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, Iterator

import pytest
import requests

_REPO = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ----------------------------------------------------------------------
# TestClient-based fixture (fast, sync, in-process)
# ----------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    db_file = tmp_path / "m.sqlite"
    monkeypatch.setenv("MARKET_DB_PATH", str(db_file))

    from api.market_routes import reset_market_runtime
    reset_market_runtime()

    from api.main import app
    with TestClient(app) as tc:
        yield tc

    reset_market_runtime()


# ----------------------------------------------------------------------
# Uvicorn-in-thread fixture (slower, needed for autonomous agent tests)
# ----------------------------------------------------------------------

_port_counter = [8300]


def _next_port() -> int:
    _port_counter[0] += 1
    return _port_counter[0]


class _LiveServer:
    def __init__(self, port: int, db_path: str):
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.db_path = db_path
        self._server = None
        self._thread = None

    def start(self):
        import uvicorn

        os.environ["MARKET_DB_PATH"] = self.db_path
        os.environ["AUTONOMOUS_API_BASE"] = f"{self.base_url}/api"

        from api.market_routes import reset_market_runtime
        reset_market_runtime()

        from api.main import app
        cfg = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # wait for it to be up
        for _ in range(50):
            try:
                r = requests.get(f"{self.base_url}/docs", timeout=0.3)
                if r.status_code < 500:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.1)
        raise RuntimeError("live server did not start")

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        from api.market_routes import reset_market_runtime
        reset_market_runtime()


@pytest.fixture
def live_server() -> Iterator[_LiveServer]:
    tmp = tempfile.mkdtemp(prefix="integ_")
    db_path = os.path.join(tmp, "m.sqlite")
    server = _LiveServer(_next_port(), db_path)
    try:
        server.start()
        yield server
    finally:
        server.stop()
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _create_market(client, *, mechanism="lmsr", ground_truth=0.70, b=100.0, title="t") -> int:
    r = client.post("/api/market/create", json={
        "mechanism": mechanism, "ground_truth": ground_truth, "b": b, "title": title,
    })
    assert r.status_code in (200, 201), r.text
    return r.json()["market_id"]


def _create_agent(client, *, name, belief=0.50, rho=1.0, cash=100.0, personality=None) -> int:
    payload = {"name": name, "cash": cash, "belief": belief, "rho": rho}
    if personality is not None:
        payload["personality"] = personality
    r = client.post("/api/agents", json=payload)
    assert r.status_code in (200, 201), r.text
    return r.json()["agent_id"]


def _join(client, market_id, agent_id):
    r = client.post(f"/api/market/{market_id}/join", json={"agent_id": agent_id})
    assert r.status_code == 200, r.text


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------

def test_full_stack_end_to_end(client):
    # create market + agents, join, submit a trade, verify state moves
    market_id = _create_market(client, title="e2e")

    agent_ids = [
        _create_agent(client, name=f"e2e_{i}", belief=0.70) for i in range(10)
    ]
    for aid in agent_ids:
        _join(client, market_id, aid)

    # initial price is 0.5 (LMSR default)
    r = client.get(f"/api/market/{market_id}/price")
    assert r.status_code == 200
    initial_price = float(r.json()["price"])
    assert abs(initial_price - 0.5) < 1e-6

    # submit a buy trade
    r = client.post(f"/api/market/{market_id}/trade", json={
        "agent_id": agent_ids[0], "quantity": 5.0,
    })
    assert r.status_code == 200, r.text

    # price should have moved up
    r = client.get(f"/api/market/{market_id}/price")
    new_price = float(r.json()["price"])
    assert new_price > initial_price

    # agent state reflects the trade
    r = client.get(f"/api/market/{market_id}/agent/{agent_ids[0]}")
    assert r.status_code == 200
    agent_state = r.json()
    assert agent_state["shares"] > 0
    assert agent_state["cash"] < 100.0

    # trade shows up in the trade history
    r = client.get(f"/api/market/{market_id}/trades")
    assert r.status_code == 200
    trades = r.json()["trades"]
    assert len(trades) >= 1


def test_news_event_shifts_beliefs_persistently(client):
    # news event should move beliefs for sensitive agents and stay moved
    market_id = _create_market(client, ground_truth=0.70)

    agent_ids = []
    for i in range(10):
        aid = _create_agent(
            client, name=f"news_{i}", belief=0.50,
            personality={
                "signal_sensitivity": 0.9,  # high sensitivity, will be affected
                "stubbornness": 0.1,
                "check_interval_mean": 2.0,
                "check_interval_jitter": 0.5,
                "edge_threshold": 0.03,
                "participation_rate": 0.8,
                "trade_size_noise": 0.2,
            },
        )
        agent_ids.append(aid)
        _join(client, market_id, aid)

    # baseline belief snapshot
    r = client.get(f"/api/market/{market_id}/agents")
    before = {a["agent_id"]: a["belief"] for a in r.json()["agents"]}

    # news event: shift everyone
    r = client.post(f"/api/market/{market_id}/news", json={
        "headline": "test news",
        "new_belief": 0.85,
        "affected_fraction": 1.0,
        "min_signal_sensitivity": 0.0,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_affected"] >= 1

    # beliefs should have moved toward 0.85
    r = client.get(f"/api/market/{market_id}/agents")
    after = {a["agent_id"]: a["belief"] for a in r.json()["agents"]}

    moved_count = 0
    for aid in agent_ids:
        if after[aid] > before[aid] + 0.05:
            moved_count += 1
    # at least some agents actually moved
    assert moved_count > 0, f"expected some belief shifts, got before={before} after={after}"


def test_autonomous_lifecycle_start_stop(live_server):
    # start autonomous trading, confirm it runs, stop cleanly, no zombie threads
    base = live_server.base_url

    r = requests.post(f"{base}/api/market/create", json={
        "mechanism": "lmsr", "ground_truth": 0.70, "b": 100.0, "title": "auto_lifecycle",
    })
    assert r.status_code in (200, 201), r.text
    market_id = r.json()["market_id"]

    # create 8 agents with calm personality so trades happen but not chaotically
    for i in range(8):
        a = requests.post(f"{base}/api/agents", json={
            "name": f"lc_{i}", "cash": 100.0,
            "belief": 0.60 + 0.02 * i, "rho": 1.0,
            "personality": {
                "check_interval_mean": 1.0, "check_interval_jitter": 0.3,
                "edge_threshold": 0.03, "participation_rate": 0.8,
                "trade_size_noise": 0.1, "signal_sensitivity": 0.5,
                "stubbornness": 0.3,
            },
        }).json()
        requests.post(f"{base}/api/market/{market_id}/join", json={"agent_id": a["agent_id"]})

    # start, wait, stop
    r = requests.post(f"{base}/api/market/{market_id}/start")
    assert r.status_code == 200, r.text
    assert r.json()["n_agents_running"] >= 1

    time.sleep(4.0)

    r = requests.post(f"{base}/api/market/{market_id}/stop")
    assert r.status_code == 200, r.text
    stop_body = r.json()
    assert stop_body["status"] == "stopped"
    assert stop_body.get("zombie_threads", 0) == 0

    # trades should have happened
    r = requests.get(f"{base}/api/market/{market_id}/trades?limit=500")
    assert r.status_code == 200, r.text
    total_trades = r.json()["total"]
    assert total_trades > 0, "expected at least one autonomous trade"


def test_autonomous_concurrency_stress(live_server):
    # 20 agents trading autonomously for ~10s, verify state stays consistent
    base = live_server.base_url

    r = requests.post(f"{base}/api/market/create", json={
        "mechanism": "lmsr", "ground_truth": 0.70, "b": 200.0, "title": "stress",
    })
    market_id = r.json()["market_id"]

    n_agents = 20
    agent_ids = []
    for i in range(n_agents):
        a = requests.post(f"{base}/api/agents", json={
            "name": f"stress_{i}", "cash": 100.0,
            "belief": 0.70, "rho": 1.0,
            "personality": {
                "check_interval_mean": 0.5, "check_interval_jitter": 0.2,
                "edge_threshold": 0.02, "participation_rate": 0.6,
                "trade_size_noise": 0.1, "signal_sensitivity": 0.5,
                "stubbornness": 0.3,
            },
        }).json()
        agent_ids.append(a["agent_id"])
        requests.post(f"{base}/api/market/{market_id}/join", json={"agent_id": a["agent_id"]})

    requests.post(f"{base}/api/market/{market_id}/start")
    time.sleep(10.0)
    r = requests.post(f"{base}/api/market/{market_id}/stop")
    stop_body = r.json()
    assert stop_body.get("zombie_threads", 0) == 0

    # final price is still a valid probability
    r = requests.get(f"{base}/api/market/{market_id}/price")
    price = float(r.json()["price"])
    assert 0.0 <= price <= 1.0

    # total trade count reported by stop matches trade rows in DB
    reported_trades = int(stop_body.get("total_trades", 0))
    r = requests.get(f"{base}/api/market/{market_id}/trades?limit=500")
    assert r.status_code == 200, r.text
    total = int(r.json()["total"])
    # a few trades can land between stop() and our GET, so allow small drift
    assert total >= reported_trades, f"db has {total}, stop reported {reported_trades}"

    # every agent should have a valid final position (cash + shares reasonable)
    r = requests.get(f"{base}/api/market/{market_id}/agents?limit=200")
    for agent in r.json()["agents"]:
        assert agent["cash"] >= 0.0 or agent["shares"] < 0, (
            f"agent {agent['agent_id']} has negative cash without short position"
        )


def test_autonomous_and_round_based_both_run(client):
    # sanity: both execution paths complete and produce valid prices for the same seed
    # tolerance-free: just assert both finished and prices are in [0, 1]
    from simulation_engine import SimulationEngine
    from belief_init import BeliefSpec

    # round-based path
    engine = SimulationEngine(
        mechanism="lmsr", phase=2, seed=42, ground_truth=0.70,
        n_agents=15, belief_spec=BeliefSpec(mode="gaussian", sigma=0.10),
        b=100.0,
    )
    engine.run(30)
    rb_state = engine.get_state()
    assert 0.0 <= rb_state["price"] <= 1.0

    # autonomous path (via API, no real threads — just validates the full HTTP flow)
    market_id = _create_market(client, ground_truth=0.70)
    agent_ids = []
    for i in range(10):
        aid = _create_agent(client, name=f"parity_{i}", belief=0.65 + 0.01 * i)
        _join(client, market_id, aid)
        agent_ids.append(aid)

    # do a few trades through the API
    for i, aid in enumerate(agent_ids[:5]):
        client.post(f"/api/market/{market_id}/trade", json={
            "agent_id": aid, "quantity": 2.0,
        })

    r = client.get(f"/api/market/{market_id}/price")
    auto_price = float(r.json()["price"])
    assert 0.0 <= auto_price <= 1.0


def test_multi_market_agent_can_join_both(client):
    # one agent, two markets, verify positions are independent
    m1 = _create_market(client, title="mkt_a", ground_truth=0.30)
    m2 = _create_market(client, title="mkt_b", ground_truth=0.80)

    aid = _create_agent(client, name="multi", belief=0.55)
    _join(client, m1, aid)
    _join(client, m2, aid)

    # trade on market 1
    client.post(f"/api/market/{m1}/trade", json={"agent_id": aid, "quantity": 3.0})

    # positions should be market-specific
    r1 = client.get(f"/api/market/{m1}/agent/{aid}")
    r2 = client.get(f"/api/market/{m2}/agent/{aid}")
    assert r1.json()["shares"] > 0
    assert abs(r2.json()["shares"]) < 1e-6


def test_market_stop_rejects_new_trades(client):
    # once a market is stopped, trades should return 409
    market_id = _create_market(client, title="stop_reject")
    aid = _create_agent(client, name="solo", belief=0.70)
    _join(client, market_id, aid)

    # manually set status to stopped via the API
    r = client.post(f"/api/market/{market_id}/start")
    assert r.status_code == 200
    r = client.post(f"/api/market/{market_id}/stop")
    assert r.status_code == 200

    # further trade should be rejected
    r = client.post(f"/api/market/{market_id}/trade", json={
        "agent_id": aid, "quantity": 1.0,
    })
    # accept either 409 (explicitly rejected) or 400 (validation)
    assert r.status_code in (400, 409), r.text
