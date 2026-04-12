"""
Integration tests for ``/api/market/*`` (persistent market + autonomous hooks).

Uses a temporary SQLite file and ``reset_market_runtime`` so each test gets a clean
service singleton. Requires ``fastapi`` and ``httpx`` (TestClient).
"""

from __future__ import annotations

import os
import sys

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
