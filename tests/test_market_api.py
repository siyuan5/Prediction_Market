"""
Integration tests for ``/api/market/*`` (persistent market + autonomous hooks).

Uses a temporary SQLite file and ``reset_market_runtime`` so each test gets a clean
service singleton. Requires ``fastapi`` and ``httpx`` (TestClient).
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

# Repo root on path for ``api`` and ``app`` imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh DB file + reset singleton so ``get_market_service`` opens this file."""
    monkeypatch.setenv("MARKET_DB_PATH", str(tmp_path / "m.sqlite"))
    from api.market_routes import reset_market_runtime

    reset_market_runtime()
    from fastapi.testclient import TestClient
    from api.main import app

    return TestClient(app)


def test_create_market_and_price(client):
    """POST /create seeds agents; GET /price returns a probability in (0,1)."""
    r = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "ground_truth": 0.66,
            "n_agents": 5,
            "initial_cash": 100.0,
            "b": 80.0,
            "seed": 1,
            "title": "test market",
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    mid = data["market_id"]
    assert data["mechanism"] == "lmsr"
    assert data["n_agents"] == 5

    pr = client.get(f"/api/market/{mid}/price")
    assert pr.status_code == 200
    body = pr.json()
    assert "price" in body
    assert 0.0 < body["price"] < 1.0
    assert body.get("best_bid") is None


def test_lmsr_trade_updates_position(client):
    """Submitting a small LMSR trade succeeds and increases total trade count."""
    cr = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "ground_truth": 0.5,
            "n_agents": 3,
            "initial_cash": 200.0,
            "b": 100.0,
            "seed": 2,
        },
    )
    mid = cr.json()["market_id"]

    ag = client.get(f"/api/market/{mid}/agents").json()
    aid = ag["agents"][0]["agent_id"]

    tr = client.post(
        f"/api/market/{mid}/trade",
        json={"agent_id": aid, "quantity": 1.0},
    )
    assert tr.status_code == 200, tr.text
    out = tr.json()
    assert "trade_id" in out
    assert out["executed_quantity"] != 0

    tf = client.get(f"/api/market/{mid}/trades").json()
    assert tf["total"] >= 1


def test_belief_shock_endpoint(client):
    """POST .../belief with new_belief updates stored belief (absolute mode)."""
    cr = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "ground_truth": 0.55,
            "n_agents": 2,
            "initial_cash": 100.0,
            "b": 50.0,
            "seed": 3,
        },
    )
    mid = cr.json()["market_id"]
    aid = client.get(f"/api/market/{mid}/agents").json()["agents"][0]["agent_id"]

    br = client.post(
        f"/api/market/{mid}/agent/{aid}/belief",
        json={"new_belief": 0.42},
    )
    assert br.status_code == 200, br.text
    b = br.json()
    assert b["new_belief"] == 0.42

    st = client.get(f"/api/market/{mid}/agent/{aid}").json()
    assert abs(st["belief"] - 0.42) < 1e-6


def test_full_market_flow_with_news_injection(client):
    """
    End-to-end agent-facing flow:
    list markets -> create -> query price -> trade -> see new price ->
    inject news -> verify affected beliefs changed in persistent state.
    """
    before = client.get("/api/markets")
    assert before.status_code == 200, before.text
    assert before.json()["total"] == 0

    created = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "title": "Flow test market",
            "ground_truth": 0.61,
            "n_agents": 6,
            "initial_cash": 150.0,
            "b": 60.0,
            "seed": 7,
            "personality_defaults": {
                "signal_sensitivity": 1.0,
                "stubbornness": 0.0,
            },
        },
    )
    assert created.status_code == 201, created.text
    cbody = created.json()
    mid = cbody["market_id"]
    assert cbody["status"] == "open"

    markets = client.get("/api/markets")
    assert markets.status_code == 200, markets.text
    mbody = markets.json()
    market_row = next(m for m in mbody["markets"] if m["market_id"] == mid)
    assert market_row["id"] == mid
    assert "trade_count_24h" in market_row
    assert "active_agents_24h" in market_row

    price_before_r = client.get(f"/api/market/{mid}/price")
    assert price_before_r.status_code == 200, price_before_r.text
    price_before = float(price_before_r.json()["price"])

    agents_before = client.get(f"/api/market/{mid}/agents")
    assert agents_before.status_code == 200, agents_before.text
    agents_before_rows = agents_before.json()["agents"]
    aid = int(agents_before_rows[0]["agent_id"])
    beliefs_before = {int(a["agent_id"]): float(a["belief"]) for a in agents_before_rows}

    trade = client.post(
        f"/api/market/{mid}/trade",
        json={"agent_id": aid, "quantity": 1.5},
    )
    assert trade.status_code == 200, trade.text
    tbody = trade.json()
    assert tbody["executed_quantity"] > 0

    price_after_r = client.get(f"/api/market/{mid}/price")
    assert price_after_r.status_code == 200, price_after_r.text
    price_after = float(price_after_r.json()["price"])
    assert price_after > price_before

    news = client.post(
        f"/api/market/{mid}/news",
        json={
            "headline": "Breaking forecast update",
            "delta": 0.20,
            "affected_fraction": 0.5,
            "min_signal_sensitivity": 0.0,
        },
    )
    assert news.status_code == 200, news.text
    nbody = news.json()
    assert nbody["market_id"] == mid
    assert nbody["n_affected"] == 3
    assert len(nbody["affected_agents"]) == 3

    affected_ids = {int(a["agent_id"]) for a in nbody["affected_agents"]}
    for row in nbody["affected_agents"]:
        assert float(row["new_belief"]) != float(row["old_belief"])
        check = client.get(f"/api/market/{mid}/agent/{row['agent_id']}")
        assert check.status_code == 200, check.text
        current = float(check.json()["belief"])
        assert abs(current - float(row["new_belief"])) < 1e-9

    agents_after = client.get(f"/api/market/{mid}/agents")
    assert agents_after.status_code == 200, agents_after.text
    for row in agents_after.json()["agents"]:
        agent_id = int(row["agent_id"])
        if agent_id in affected_ids:
            continue
        assert abs(float(row["belief"]) - beliefs_before[agent_id]) < 1e-9


def test_two_market_start_stop_lifecycle_no_zombies(client, monkeypatch):
    class FakeAutonomousAgent:
        def __init__(self, agent_id, api_base_url, personality, belief, rho, cash):
            self.agent_id = int(agent_id)
            self._stop = threading.Event()
            self._tick = 0

        def run(self):
            from api.market_routes import get_market_service

            svc = get_market_service()
            while not self._stop.wait(0.01):
                running = sorted(svc.list_markets(status="running"), key=lambda m: int(m["id"]))
                if not running:
                    continue
                target = running[self._tick % len(running)]
                self._tick += 1
                try:
                    svc.execute_lmsr_trade(int(target["id"]), self.agent_id, quantity=0.02)
                except Exception:
                    continue

        def stop(self):
            self._stop.set()

    monkeypatch.setattr("agent_runner.AutonomousAgent", FakeAutonomousAgent)

    m1 = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "title": "M1",
            "ground_truth": 0.62,
            "n_agents": 25,
            "initial_cash": 100.0,
            "b": 70.0,
            "seed": 10,
        },
    ).json()["market_id"]
    m2 = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "title": "M2",
            "ground_truth": 0.55,
            "n_agents": 25,
            "initial_cash": 100.0,
            "b": 70.0,
            "seed": 11,
        },
    ).json()["market_id"]

    s1 = client.post(f"/api/market/{m1}/start")
    s2 = client.post(f"/api/market/{m2}/start")
    assert s1.status_code == 200, s1.text
    assert s2.status_code == 200, s2.text
    assert s1.json()["n_agents_running"] == 25
    assert s2.json()["n_agents_running"] == 25

    time.sleep(0.6)
    t1_before = client.get(f"/api/market/{m1}/trades").json()["total"]
    t2_before = client.get(f"/api/market/{m2}/trades").json()["total"]
    assert t1_before > 0
    assert t2_before > 0

    stop1 = client.post(f"/api/market/{m1}/stop")
    assert stop1.status_code == 200, stop1.text
    assert stop1.json()["zombie_threads"] == 0

    time.sleep(0.4)
    t2_after = client.get(f"/api/market/{m2}/trades").json()["total"]
    assert t2_after > t2_before

    stop2 = client.post(f"/api/market/{m2}/stop")
    assert stop2.status_code == 200, stop2.text
    assert stop2.json()["zombie_threads"] == 0

    deadline = time.time() + 2.0
    while time.time() < deadline:
        alive = [t for t in threading.enumerate() if t.name.startswith("agent-runner-")]
        if not alive:
            break
        time.sleep(0.05)
    assert [
        t.name for t in threading.enumerate() if t.name.startswith("agent-runner-")
    ] == []
