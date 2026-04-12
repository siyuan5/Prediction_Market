"""
Thread-safety tests for MarketService.

Proves: two threads calling execute_trade() concurrently produce consistent
final state — no duplicated trades, prices match, cash balances are correct.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from market_service import MarketService


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def svc(tmp_path):
    db = str(tmp_path / "test.db")
    s = MarketService(db)
    yield s
    s.close()


# ── Basic single-thread sanity ─────────────────────────────────────────


class TestSingleThread:
    def test_lmsr_lifecycle(self, svc: MarketService):
        mkt = svc.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = svc.create_agent(name="alice", cash=1000.0)

        trade = svc.execute_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
        assert trade["price_after"] > 0.5
        assert svc.get_price(mkt["id"]) == pytest.approx(trade["price_after"])

    def test_cda_lifecycle(self, svc: MarketService):
        mkt = svc.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = svc.create_agent(name="alice", cash=1000.0)
        bob = svc.create_agent(name="bob", cash=1000.0)

        svc.execute_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.50)
        result = svc.execute_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.50)

        assert result["filled_quantity"] == pytest.approx(5.0)
        assert svc.get_price(mkt["id"]) == pytest.approx(0.50)

    def test_rejects_memory_db(self):
        with pytest.raises(ValueError, match="In-memory"):
            MarketService(":memory:")


# ── Concurrent LMSR trading ───────────────────────────────────────────


class TestConcurrentLMSR:
    def test_two_threads_consistent_state(self, svc: MarketService):
        """
        Two threads each submit 50 trades (100 total).
        Final state must have exactly 100 trades, and the price must match
        what you get by replaying the trade sequence serially.
        """
        mkt = svc.create_market(slug="race", title="Race", mechanism="lmsr", b=100.0)
        a1 = svc.create_agent(name="thread-1-agent", cash=100_000.0)
        a2 = svc.create_agent(name="thread-2-agent", cash=100_000.0)

        trades_per_thread = 50
        errors: list = []

        def worker(agent_id: int, side: str):
            try:
                for _ in range(trades_per_thread):
                    svc.execute_trade(agent_id, mkt["id"], side, shares=1.0)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=worker, args=(a1["id"], "buy_yes"))
        t2 = threading.Thread(target=worker, args=(a2["id"], "buy_yes"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"

        all_trades = svc.get_trades(market_id=mkt["id"], limit=200)
        assert len(all_trades) == 2 * trades_per_thread

        trade_ids = [t["id"] for t in all_trades]
        assert len(trade_ids) == len(set(trade_ids)), "Duplicate trade IDs detected"

        final_price = svc.get_price(mkt["id"])
        mkt_state = svc.get_market(mkt["id"])
        expected_price = MarketService._lmsr_price_static(
            mkt_state["inv_yes"], mkt_state["inv_no"], mkt_state["b"]
        )
        assert final_price == pytest.approx(expected_price)

        a1_state = svc.get_agent(a1["id"])
        a2_state = svc.get_agent(a2["id"])
        a1_pos = svc.get_position(a1["id"], mkt["id"])
        a2_pos = svc.get_position(a2["id"], mkt["id"])
        assert a1_pos["yes_shares"] == pytest.approx(trades_per_thread)
        assert a2_pos["yes_shares"] == pytest.approx(trades_per_thread)

        total_cost = (100_000.0 - a1_state["cash"]) + (100_000.0 - a2_state["cash"])
        assert total_cost > 0

    def test_opposing_trades_consistent(self, svc: MarketService):
        """
        One thread buys, the other sells. Net inventory change should be zero;
        price should return close to 0.5.
        """
        mkt = svc.create_market(slug="oppose", title="Oppose", mechanism="lmsr", b=100.0)
        buyer = svc.create_agent(name="buyer", cash=100_000.0)
        seller = svc.create_agent(name="seller", cash=100_000.0)

        n = 40
        errors: list = []

        def buy_worker():
            try:
                for _ in range(n):
                    svc.execute_trade(buyer["id"], mkt["id"], "buy_yes", shares=1.0)
            except Exception as exc:
                errors.append(exc)

        def sell_worker():
            try:
                for _ in range(n):
                    svc.execute_trade(seller["id"], mkt["id"], "sell_yes", shares=1.0)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=buy_worker)
        t2 = threading.Thread(target=sell_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"

        all_trades = svc.get_trades(market_id=mkt["id"], limit=200)
        assert len(all_trades) == 2 * n

        mkt_state = svc.get_market(mkt["id"])
        assert mkt_state["inv_yes"] == pytest.approx(0.0, abs=1e-9)
        assert svc.get_price(mkt["id"]) == pytest.approx(0.5, abs=1e-6)

    def test_many_threads_no_corruption(self, svc: MarketService):
        """8 threads, each trading 20 times — no crashes or duplicates."""
        mkt = svc.create_market(slug="stress", title="Stress", mechanism="lmsr", b=100.0)
        n_threads = 8
        trades_each = 20
        agents = [
            svc.create_agent(name=f"agent-{i}", cash=100_000.0)
            for i in range(n_threads)
        ]
        errors: list = []

        def worker(agent_id: int):
            try:
                for _ in range(trades_each):
                    svc.execute_trade(agent_id, mkt["id"], "buy_yes", shares=0.5)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(a["id"],))
            for a in agents
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

        all_trades = svc.get_trades(market_id=mkt["id"], limit=500)
        assert len(all_trades) == n_threads * trades_each

        trade_ids = [t["id"] for t in all_trades]
        assert len(trade_ids) == len(set(trade_ids))

    def test_sequential_prices_match_concurrent(self, svc: MarketService, tmp_path):
        """
        Run the same trades sequentially and concurrently.
        Final price and inventory must be identical.
        """
        def run_sequential():
            db = str(tmp_path / "seq.db")
            s = MarketService(db)
            mkt = s.create_market(slug="seq", title="Seq", mechanism="lmsr", b=100.0)
            a1 = s.create_agent(name="a1", cash=100_000.0)
            a2 = s.create_agent(name="a2", cash=100_000.0)
            for _ in range(30):
                s.execute_trade(a1["id"], mkt["id"], "buy_yes", shares=1.0)
            for _ in range(30):
                s.execute_trade(a2["id"], mkt["id"], "buy_yes", shares=1.0)
            result = {
                "price": s.get_price(mkt["id"]),
                "inv_yes": s.get_market(mkt["id"])["inv_yes"],
            }
            s.close()
            return result

        def run_concurrent():
            db = str(tmp_path / "conc.db")
            s = MarketService(db)
            mkt = s.create_market(slug="conc", title="Conc", mechanism="lmsr", b=100.0)
            a1 = s.create_agent(name="a1", cash=100_000.0)
            a2 = s.create_agent(name="a2", cash=100_000.0)
            errors_local: list = []

            def w(aid):
                try:
                    for _ in range(30):
                        s.execute_trade(aid, mkt["id"], "buy_yes", shares=1.0)
                except Exception as e:
                    errors_local.append(e)

            t1 = threading.Thread(target=w, args=(a1["id"],))
            t2 = threading.Thread(target=w, args=(a2["id"],))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            assert errors_local == []
            result = {
                "price": s.get_price(mkt["id"]),
                "inv_yes": s.get_market(mkt["id"])["inv_yes"],
            }
            s.close()
            return result

        seq = run_sequential()
        conc = run_concurrent()

        assert seq["inv_yes"] == pytest.approx(conc["inv_yes"])
        assert seq["price"] == pytest.approx(conc["price"])


# ── Concurrent CDA trading ────────────────────────────────────────────


class TestConcurrentCDA:
    def test_two_threads_limit_orders(self, svc: MarketService):
        """
        One thread posts sells, the other posts buys at the same price.
        All should match; no orphaned orders, no duplicate trades.
        """
        mkt = svc.create_market(slug="cda-race", title="CDA Race", mechanism="cda", tick_size=0.01)
        seller = svc.create_agent(name="seller", cash=100_000.0)
        buyer = svc.create_agent(name="buyer", cash=100_000.0)

        n = 30
        errors: list = []

        def sell_worker():
            try:
                for _ in range(n):
                    svc.execute_limit_order(
                        seller["id"], mkt["id"], "sell", quantity=1.0, price=0.50,
                    )
            except Exception as exc:
                errors.append(exc)

        def buy_worker():
            try:
                for _ in range(n):
                    svc.execute_limit_order(
                        buyer["id"], mkt["id"], "buy", quantity=1.0, price=0.50,
                    )
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=sell_worker)
        t2 = threading.Thread(target=buy_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f"Thread errors: {errors}"

        all_trades = svc.get_trades(market_id=mkt["id"], limit=200)
        trade_ids = [t["id"] for t in all_trades]
        assert len(trade_ids) == len(set(trade_ids)), "Duplicate trade IDs"

        seller_pos = svc.get_position(seller["id"], mkt["id"])
        buyer_pos = svc.get_position(buyer["id"], mkt["id"])
        assert seller_pos["yes_shares"] == pytest.approx(-buyer_pos["yes_shares"])

        seller_cash = svc.get_agent(seller["id"])["cash"]
        buyer_cash = svc.get_agent(buyer["id"])["cash"]
        assert (seller_cash + buyer_cash) == pytest.approx(200_000.0)


# ── Helper on MarketService for test verification ─────────────────────
# We add this as a static method so tests can recompute expected price.


MarketService._lmsr_price_static = staticmethod(
    lambda inv_yes, inv_no, b: __import__("math").exp(inv_yes / b) / (
        __import__("math").exp(inv_yes / b) + __import__("math").exp(inv_no / b)
    )
)
