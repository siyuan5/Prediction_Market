"""
Integration tests for ``/api/market/*`` + ``/api/agents*``.

Uses a temporary SQLite file and ``reset_market_runtime`` so each test gets a
clean service singleton.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def client(monkeypatch):
    tmp_base = os.path.join(ROOT, "tmp_market_api_tests", str(uuid.uuid4()))
    os.makedirs(tmp_base, exist_ok=True)
    monkeypatch.setenv("MARKET_DB_PATH", os.path.join(tmp_base, "m.sqlite"))
    from api.market_routes import reset_market_runtime

    reset_market_runtime()
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as tc:
        yield tc
    reset_market_runtime()
    shutil.rmtree(tmp_base, ignore_errors=True)


def _create_agent(client, *, name: str, belief: float = 0.6):
    r = client.post(
        "/api/agents",
        json={
            "name": name,
            "cash": 150.0,
            "belief": belief,
            "rho": 1.2,
            "personality": {"signal_sensitivity": 1.0, "stubbornness": 0.0},
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_create_market_and_price_no_agent_spawn(client):
    r = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "ground_truth": 0.66,
            "b": 80.0,
            "title": "test market",
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    mid = data["market_id"]
    assert data["mechanism"] == "lmsr"

    ag = client.get(f"/api/market/{mid}/agents").json()
    assert ag["total"] == 0

    pr = client.get(f"/api/market/{mid}/price")
    assert pr.status_code == 200
    body = pr.json()
    assert "price" in body
    assert 0.0 < body["price"] < 1.0
    assert body.get("best_bid") is None
    assert body.get("ground_truth") == pytest.approx(0.66)
    assert body.get("mean_belief") is None


def test_delete_market_removes_row_and_children(client):
    r = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "ground_truth": 0.55,
            "b": 50.0,
            "title": "to delete",
        },
    )
    assert r.status_code == 201, r.text
    mid = r.json()["market_id"]
    aid = _create_agent(client, name="deltrader", belief=0.5)["agent_id"]
    client.post(f"/api/market/{mid}/join", json={"agent_id": aid})
    tr = client.post(f"/api/market/{mid}/trade", json={"agent_id": aid, "quantity": 1.0})
    assert tr.status_code == 200, tr.text

    d = client.delete(f"/api/market/{mid}")
    assert d.status_code == 200, d.text
    body = d.json()
    assert body["deleted"] is True
    assert body["market_id"] == mid
    assert body.get("agents_removed") == 1

    assert client.get(f"/api/market/{mid}/detail").status_code == 404
    listed = client.get("/api/markets?status=all&limit=200&offset=0")
    assert listed.status_code == 200
    ids = [m.get("market_id") or m.get("id") for m in listed.json().get("markets", [])]
    assert mid not in ids
    pool = client.get("/api/agents?limit=500&offset=0")
    assert pool.status_code == 200
    assert not any(a["agent_id"] == aid for a in pool.json().get("agents", []))


def test_delete_market_keeps_agent_with_second_market(client):
    r1 = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "ground_truth": 0.5, "b": 40.0, "title": "m1"},
    )
    r2 = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "ground_truth": 0.5, "b": 40.0, "title": "m2"},
    )
    assert r1.status_code == 201 and r2.status_code == 201
    mid1 = r1.json()["market_id"]
    mid2 = r2.json()["market_id"]
    aid = _create_agent(client, name="two-market-trader", belief=0.55)["agent_id"]
    assert client.post(f"/api/market/{mid1}/join", json={"agent_id": aid}).status_code == 200
    assert client.post(f"/api/market/{mid2}/join", json={"agent_id": aid}).status_code == 200

    d = client.delete(f"/api/market/{mid1}")
    assert d.status_code == 200, d.text
    assert d.json().get("agents_removed") == 0

    pool = client.get("/api/agents?limit=500&offset=0").json()
    assert any(a["agent_id"] == aid for a in pool["agents"])


def test_price_mean_belief_is_market_scoped(client):
    _create_agent(client, name="m1", belief=0.5)
    _create_agent(client, name="m2", belief=0.7)
    r = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "ground_truth": 0.72,
            "b": 80.0,
            "title": "mean belief test",
        },
    )
    assert r.status_code == 201, r.text
    mid = r.json()["market_id"]
    pr = client.get(f"/api/market/{mid}/price")
    assert pr.status_code == 200
    body = pr.json()
    assert body["ground_truth"] == pytest.approx(0.72)
    assert body["mean_belief"] is None


def test_delete_agent_hides_from_ui_but_keeps_trade_history(client):
    aid = _create_agent(client, name="soft-delete-me", belief=0.62)["agent_id"]
    mid = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "ground_truth": 0.6, "b": 70.0, "title": "agent-delete-history"},
    ).json()["market_id"]
    assert client.post(f"/api/market/{mid}/join", json={"agent_id": aid}).status_code == 200
    tr = client.post(f"/api/market/{mid}/trade", json={"agent_id": aid, "quantity": 1.0})
    assert tr.status_code == 200, tr.text

    d = client.delete(f"/api/agents/{aid}")
    assert d.status_code == 200, d.text
    body = d.json()
    assert body["deleted"] is True
    assert body["agent_id"] == aid
    assert body["trade_count_retained"] >= 1

    listed = client.get("/api/agents?limit=500&offset=0").json()["agents"]
    assert not any(a["agent_id"] == aid for a in listed)
    assert client.get(f"/api/agents/{aid}").status_code == 404
    assert client.post(f"/api/market/{mid}/trade", json={"agent_id": aid, "quantity": 0.5}).status_code == 404

    # Trade rows remain in DB and are still visible in market history.
    mtr = client.get(f"/api/market/{mid}/trades?limit=500")
    assert mtr.status_code == 200, mtr.text
    rows = mtr.json()["trades"]
    assert any(int(r["agent_id"]) == aid for r in rows)


def test_agents_create_list_patch_and_alias(client):
    created = _create_agent(client, name="alice", belief=0.61)
    assert created["name"] == "alice"
    assert created["belief"] == pytest.approx(0.61)

    alias = client.post(
        "/api/agents/create",
        json={"name": "bob", "cash": 200.0, "belief": 0.45, "rho": 1.0},
    )
    assert alias.status_code == 201, alias.text

    listed = client.get("/api/agents?limit=10&offset=0")
    assert listed.status_code == 200, listed.text
    rows = listed.json()["agents"]
    assert listed.json()["total"] == 2
    assert {r["name"] for r in rows} == {"alice", "bob"}

    patched = client.patch(
        f"/api/agents/{created['agent_id']}",
        json={"belief": 0.72, "cash": 180.0},
    )
    assert patched.status_code == 200, patched.text
    p = patched.json()
    assert p["belief"] == pytest.approx(0.72)
    assert p["cash"] == pytest.approx(180.0)


def test_market_join_and_lmsr_trade(client):
    aid = _create_agent(client, name="trader")["agent_id"]
    mid = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "ground_truth": 0.5, "b": 100.0},
    ).json()["market_id"]

    j = client.post(f"/api/market/{mid}/join", json={"agent_id": aid})
    assert j.status_code == 200, j.text
    assert j.json()["status"] == "joined"

    tr = client.post(
        f"/api/market/{mid}/trade",
        json={"agent_id": aid, "quantity": 1.0},
    )
    assert tr.status_code == 200, tr.text
    out = tr.json()
    assert out["trade_id"] is not None
    assert out["executed_quantity"] != 0


def test_full_market_flow_with_news_injection(client):
    before = client.get("/api/markets")
    assert before.status_code == 200, before.text
    assert before.json()["total"] == 0

    created_agents = [
        _create_agent(client, name=f"flow-a{i}", belief=0.55 + i * 0.01)
        for i in range(6)
    ]

    created = client.post(
        "/api/market/create",
        json={
            "mechanism": "lmsr",
            "title": "Flow test market",
            "ground_truth": 0.61,
            "b": 60.0,
        },
    )
    assert created.status_code == 201, created.text
    mid = created.json()["market_id"]
    for row in created_agents:
        assert client.post(
            f"/api/market/{mid}/join",
            json={"agent_id": int(row["agent_id"])},
        ).status_code == 200

    markets = client.get("/api/markets")
    assert markets.status_code == 200, markets.text
    mbody = markets.json()
    market_row = next(m for m in mbody["markets"] if m["market_id"] == mid)
    assert market_row["id"] == mid
    assert "trade_count_24h" in market_row
    assert "active_agents_24h" in market_row

    price_before = float(client.get(f"/api/market/{mid}/price").json()["price"])
    aid = int(created_agents[0]["agent_id"])
    trade = client.post(
        f"/api/market/{mid}/trade",
        json={"agent_id": aid, "quantity": 1.5},
    )
    assert trade.status_code == 200, trade.text
    assert trade.json()["executed_quantity"] > 0

    price_after = float(client.get(f"/api/market/{mid}/price").json()["price"])
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

    for row in nbody["affected_agents"]:
        check = client.get(f"/api/market/{mid}/agent/{row['agent_id']}")
        assert check.status_code == 200, check.text
        current = float(check.json()["belief"])
        assert abs(current - float(row["new_belief"])) < 1e-9


def test_news_events_persist_and_history_endpoint_returns_them(client):
    for i in range(4):
        _create_agent(client, name=f"news-a{i}", belief=0.55 + i * 0.01)
    mid = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "title": "News history market", "ground_truth": 0.63, "b": 55.0},
    ).json()["market_id"]

    n1 = client.post(
        f"/api/market/{mid}/news",
        json={
            "headline": "Event one",
            "delta": 0.10,
            "affected_fraction": 0.5,
            "min_signal_sensitivity": 0.0,
        },
    )
    n2 = client.post(
        f"/api/market/{mid}/news",
        json={
            "headline": "Event two",
            "new_belief": 0.72,
            "affected_fraction": 0.5,
            "min_signal_sensitivity": 0.0,
        },
    )
    assert n1.status_code == 200, n1.text
    assert n2.status_code == 200, n2.text
    assert n1.json()["news_event_id"] != n2.json()["news_event_id"]

    hist = client.get(f"/api/market/{mid}/news?limit=10&offset=0")
    assert hist.status_code == 200, hist.text
    body = hist.json()
    assert body["total"] >= 2
    events = body["events"]
    headlines = [e["headline"] for e in events]
    assert "Event one" in headlines
    assert "Event two" in headlines
    for ev in events:
        assert "at_timestamp" in ev
        assert ev["mode"] in ("absolute", "delta")


def test_two_market_start_stop_lifecycle_no_zombies(client, monkeypatch):
    class FakeAutonomousAgent:
        def __init__(self, agent_id, api_base_url, personality, belief, rho, cash, **kwargs):
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

    for i in range(50):
        _create_agent(client, name=f"auto-{i}", belief=0.6)

    m1 = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "title": "M1", "ground_truth": 0.62, "b": 70.0},
    ).json()["market_id"]
    m2 = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "title": "M2", "ground_truth": 0.55, "b": 70.0},
    ).json()["market_id"]

    s1 = client.post(f"/api/market/{m1}/start")
    s2 = client.post(f"/api/market/{m2}/start")
    assert s1.status_code == 200, s1.text
    assert s2.status_code == 200, s2.text
    assert s1.json()["n_agents_running"] == 50
    assert s2.json()["n_agents_running"] == 50

    time.sleep(0.6)
    t1_before = client.get(f"/api/market/{m1}/trades").json()["total"]
    t2_before = client.get(f"/api/market/{m2}/trades").json()["total"]
    assert t1_before > 0
    assert t2_before > 0

    stop1 = client.post(f"/api/market/{m1}/stop")
    assert stop1.status_code == 200, stop1.text
    assert stop1.json()["zombie_threads"] == 0

    running = client.get("/api/markets?status=running").json()["markets"]
    running_ids = {int(m["market_id"]) for m in running}
    assert m1 not in running_ids
    assert m2 in running_ids

    duplicate_start = client.post(f"/api/market/{m2}/start")
    assert duplicate_start.status_code == 409

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


def test_market_detail_and_comment_tick(client):
    mid = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "title": "Detail test", "ground_truth": 0.6, "b": 50.0},
    ).json()["market_id"]
    d = client.get(f"/api/market/{mid}/detail")
    assert d.status_code == 200, d.text
    body = d.json()
    assert body["market_id"] == mid
    assert body["title"] == "Detail test"
    assert body["mechanism"] == "lmsr"
    assert "price" in body

    aid = _create_agent(client, name="c1", belief=0.55)["agent_id"]
    client.post(f"/api/market/{mid}/join", json={"agent_id": aid})
    client.post(
        f"/api/market/{mid}/trade",
        json={"agent_id": aid, "quantity": 2.0},
    )
    tick = client.post(f"/api/market/{mid}/comments/tick")
    assert tick.status_code == 200, tick.text
    assert tick.json()["appended"] == 1
    cm = client.get(f"/api/market/{mid}/comments?since=0")
    assert cm.status_code == 200, cm.text
    assert cm.json()["total"] >= 1


def test_agent_detail_activity_endpoints(client):
    aid = _create_agent(client, name="profiled-agent", belief=0.58)["agent_id"]
    mid = client.post(
        "/api/market/create",
        json={"mechanism": "lmsr", "title": "Agent activity market", "ground_truth": 0.6, "b": 50.0},
    ).json()["market_id"]
    assert client.post(f"/api/market/{mid}/join", json={"agent_id": aid}).status_code == 200
    trade = client.post(f"/api/market/{mid}/trade", json={"agent_id": aid, "quantity": 2.0})
    assert trade.status_code == 200, trade.text
    assert client.post(f"/api/market/{mid}/comments/tick").status_code == 200

    profile = client.get(f"/api/agents/{aid}")
    assert profile.status_code == 200, profile.text
    assert profile.json()["name"] == "profiled-agent"

    market_agent = client.get(f"/api/market/{mid}/agent/{aid}").json()
    markets = client.get(f"/api/agents/{aid}/markets")
    assert markets.status_code == 200, markets.text
    market_rows = markets.json()["markets"]
    assert len(market_rows) == 1
    assert market_rows[0]["market_id"] == mid
    assert market_rows[0]["title"] == "Agent activity market"
    assert market_rows[0]["trade_count"] == 1
    assert market_rows[0]["unrealized_pnl"] == pytest.approx(market_agent["pnl"])

    trades = client.get(f"/api/agents/{aid}/trades")
    assert trades.status_code == 200, trades.text
    trade_rows = trades.json()["trades"]
    assert len(trade_rows) == 1
    assert trade_rows[0]["market_id"] == mid
    assert trade_rows[0]["market_title"] == "Agent activity market"
    assert trade_rows[0]["quantity"] == pytest.approx(2.0)

    comments = client.get(f"/api/agents/{aid}/comments")
    assert comments.status_code == 200, comments.text
    comment_rows = comments.json()["comments"]
    assert len(comment_rows) == 1
    assert comment_rows[0]["market_id"] == mid
    assert comment_rows[0]["text"]
