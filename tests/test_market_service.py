"""
Thread-safety tests for MarketService.

Proves:
- 10 concurrent execute_lmsr_trade calls produce consistent final state
- CDA order matching works with crossing orders
- Clipping works when agent has insufficient cash
- get_price_snapshot returns correct dict
"""

from __future__ import annotations

import math
import sys
import threading
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from market_service import MarketService
from market_store import MarketStore


def _lmsr_price(inv_yes: float, inv_no: float, b: float) -> float:
    e_yes = math.exp(inv_yes / b)
    e_no = math.exp(inv_no / b)
    return e_yes / (e_yes + e_no)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def svc(tmp_path):
    db = str(tmp_path / "test.db")
    s = MarketService(db)
    yield s
    s.close()


def _make_open_lmsr(svc, slug="test", b=100.0, **kw):
    mkt = svc.create_market(slug=slug, title=slug, mechanism="lmsr", b=b, **kw)
    svc.set_market_status(mkt["id"], "open")
    return svc.get_market(mkt["id"])


def _make_running_lmsr(svc, slug="test", b=100.0, **kw):
    mkt = svc.create_market(slug=slug, title=slug, mechanism="lmsr", b=b, **kw)
    svc.set_market_status(mkt["id"], "running")
    return svc.get_market(mkt["id"])


def _make_open_cda(svc, slug="test", tick_size=0.01, **kw):
    mkt = svc.create_market(slug=slug, title=slug, mechanism="cda", tick_size=tick_size, **kw)
    svc.set_market_status(mkt["id"], "open")
    return svc.get_market(mkt["id"])


# ── Basic single-thread sanity ─────────────────────────────────────────


class TestSingleThread:
    def test_lmsr_lifecycle(self, svc: MarketService):
        mkt = _make_open_lmsr(svc)
        agent = svc.create_agent(name="alice", cash=1000.0)
        trade = svc.execute_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
        assert trade["price_after"] > 0.5
        assert svc.get_price(mkt["id"]) == pytest.approx(trade["price_after"])

    def test_cda_lifecycle(self, svc: MarketService):
        mkt = _make_open_cda(svc)
        alice = svc.create_agent(name="alice", cash=1000.0)
        bob = svc.create_agent(name="bob", cash=1000.0)
        svc.execute_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.50)
        result = svc.execute_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.50)
        assert result["filled_quantity"] == pytest.approx(5.0)
        assert svc.get_price(mkt["id"]) == pytest.approx(0.50)

    def test_rejects_memory_db(self):
        with pytest.raises(ValueError, match="In-memory"):
            MarketService(":memory:")

    def test_market_status_lifecycle(self, svc: MarketService):
        mkt = svc.create_market(slug="s", title="S", mechanism="lmsr", b=100.0, ground_truth=0.7)
        assert mkt["status"] == "created"
        assert mkt["ground_truth"] == 0.7
        updated = svc.set_market_status(mkt["id"], "running")
        assert updated["status"] == "running"

    def test_agent_with_market_fields(self, svc: MarketService):
        mkt = svc.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = svc.create_agent(
            name="alice", cash=1000.0, market_id=mkt["id"],
            belief=0.6, rho=1.5, personality="aggressive",
        )
        assert agent["belief"] == 0.6
        fetched = svc.get_agent(agent["id"], market_id=mkt["id"])
        assert fetched["belief"] == 0.6
        assert fetched["rho"] == 1.5

    def test_set_agent_belief(self, svc: MarketService):
        mkt = svc.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = svc.create_agent(name="bob", cash=500.0, market_id=mkt["id"], belief=0.3)
        old = svc.set_agent_belief(mkt["id"], agent["id"], 0.9)
        assert old == 0.3
        pos = svc.get_position(agent["id"], mkt["id"])
        assert pos["belief"] == 0.9

    def test_update_agent_portfolio(self, svc: MarketService):
        mkt = _make_open_lmsr(svc)
        agent = svc.create_agent(name="alice", cash=1000.0)
        result = svc.update_agent_portfolio(mkt["id"], agent["id"], cash_delta=-50.0, shares_delta=10.0)
        assert result["cash"] == pytest.approx(950.0)
        assert result["yes_shares"] == pytest.approx(10.0)

    def test_get_trades_since_trade_id(self, svc: MarketService):
        mkt = _make_open_lmsr(svc)
        agent = svc.create_agent(name="alice", cash=10000.0)
        t1 = svc.execute_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)
        t2 = svc.execute_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)
        trades = svc.get_trades(market_id=mkt["id"], since_trade_id=t1["trade_id"])
        assert len(trades) == 1
        assert trades[0]["id"] == t2["trade_id"]

    def test_cannot_trade_created_market(self, svc: MarketService):
        mkt = svc.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = svc.create_agent(name="alice", cash=1000.0)
        with pytest.raises(ValueError, match="trading requires"):
            svc.execute_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)


# ── execute_lmsr_trade (new API) ──────────────────────────────────────


class TestExecuteLMSRTrade:
    def test_buy_positive_quantity(self, svc: MarketService):
        mkt = _make_open_lmsr(svc, slug="lmsr-pos")
        agent = svc.create_agent(name="alice", cash=10000.0)
        result = svc.execute_lmsr_trade(mkt["id"], agent["id"], quantity=5.0)
        assert result["trade_id"] is not None
        assert result["quantity"] == pytest.approx(5.0)
        assert result["clipped"] is False
        assert result["price_after"] > result["price_before"]

    def test_sell_negative_quantity(self, svc: MarketService):
        mkt = _make_open_lmsr(svc, slug="lmsr-neg")
        agent = svc.create_agent(name="bob", cash=10000.0)
        result = svc.execute_lmsr_trade(mkt["id"], agent["id"], quantity=-5.0)
        assert result["quantity"] == pytest.approx(-5.0)
        assert result["price_after"] < result["price_before"]
        assert result["cost"] < 0

    def test_zero_quantity_rejected(self, svc: MarketService):
        mkt = _make_open_lmsr(svc, slug="lmsr-zero")
        agent = svc.create_agent(name="alice", cash=1000.0)
        with pytest.raises(ValueError, match="non-zero"):
            svc.execute_lmsr_trade(mkt["id"], agent["id"], quantity=0)


# ── Clipping (insufficient cash) ──────────────────────────────────────


class TestClipping:
    def test_lmsr_clips_buy_to_affordable(self, svc: MarketService):
        """Agent with small cash tries a big buy; quantity is clipped, not rejected."""
        mkt = _make_open_lmsr(svc, slug="clip-lmsr")
        agent = svc.create_agent(name="pooralice", cash=1.0)
        result = svc.execute_lmsr_trade(mkt["id"], agent["id"], quantity=1000.0)

        assert result["clipped"] is True
        assert 0 < result["quantity"] < 1000.0
        assert result["cost"] <= 1.0 + 1e-9

        agent_after = svc.get_agent(agent["id"])
        assert agent_after["cash"] >= -1e-9

        pos = svc.get_position(agent["id"], mkt["id"])
        assert pos["yes_shares"] == pytest.approx(result["quantity"])

    def test_lmsr_clips_to_zero_when_broke(self, svc: MarketService):
        """Agent with nearly zero cash gets clipped to zero -> no trade."""
        mkt = _make_open_lmsr(svc, slug="clip-zero")
        agent = svc.create_agent(name="broke", cash=0.0)
        result = svc.execute_lmsr_trade(mkt["id"], agent["id"], quantity=10.0)
        assert result["clipped"] is True
        assert result["quantity"] == 0.0
        assert result["trade_id"] is None
        assert svc.get_price(mkt["id"]) == pytest.approx(0.5)

    def test_sell_never_clips(self, svc: MarketService):
        """Sells produce negative cost (agent receives cash), no clipping needed."""
        mkt = _make_open_lmsr(svc, slug="clip-sell")
        agent = svc.create_agent(name="seller", cash=0.0)
        result = svc.execute_lmsr_trade(mkt["id"], agent["id"], quantity=-10.0)
        assert result["clipped"] is False
        assert result["quantity"] == pytest.approx(-10.0)
        assert result["cost"] < 0
        assert svc.get_agent(agent["id"])["cash"] > 0

    def test_execute_trade_wrapper_raises_on_full_clip(self, svc: MarketService):
        """The backward-compat execute_trade raises ValueError when clip -> zero."""
        mkt = _make_open_lmsr(svc, slug="clip-compat")
        agent = svc.create_agent(name="broke", cash=0.0)
        with pytest.raises(ValueError, match="Insufficient funds"):
            svc.execute_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)

    def test_cda_stops_on_insufficient_buyer_cash(self, svc: MarketService):
        """CDA buyer with limited cash gets partial fill clipped to affordable qty."""
        mkt = _make_open_cda(svc, slug="cda-clip")
        seller = svc.create_agent(name="seller", cash=100_000.0)
        buyer = svc.create_agent(name="buyer", cash=0.55)

        svc.execute_cda_order(mkt["id"], seller["id"], "sell", 10.0, 0.50, "limit")
        result = svc.execute_cda_order(mkt["id"], buyer["id"], "buy", 10.0, 0.50, "limit")

        assert result["filled_quantity"] == pytest.approx(1.1)
        assert result["remaining_quantity"] == pytest.approx(8.9)

        assert svc.get_agent(buyer["id"])["cash"] == pytest.approx(0.0, abs=1e-9)


# ── execute_cda_order (new API) ───────────────────────────────────────


class TestExecuteCDAOrder:
    def test_limit_order_rests(self, svc: MarketService):
        mkt = _make_open_cda(svc, slug="cda-rest")
        agent = svc.create_agent(name="alice", cash=1000.0)
        result = svc.execute_cda_order(mkt["id"], agent["id"], "buy", 5.0, 0.40, "limit")
        assert result["filled_quantity"] == pytest.approx(0.0)
        assert result["resting_order_id"] is not None
        book = svc.get_order_book(mkt["id"])
        assert len(book["bids"]) == 1

    def test_crossing_orders_match(self, svc: MarketService):
        mkt = _make_open_cda(svc, slug="cda-cross")
        alice = svc.create_agent(name="alice", cash=1000.0)
        bob = svc.create_agent(name="bob", cash=1000.0)

        svc.execute_cda_order(mkt["id"], alice["id"], "sell", 10.0, 0.55, "limit")
        result = svc.execute_cda_order(mkt["id"], bob["id"], "buy", 10.0, 0.55, "limit")

        assert result["filled_quantity"] == pytest.approx(10.0)
        assert len(result["trades"]) == 1
        assert result["trades"][0]["price"] == pytest.approx(0.55)

        notional = 0.55 * 10.0
        assert svc.get_agent(bob["id"])["cash"] == pytest.approx(1000.0 - notional)
        assert svc.get_agent(alice["id"])["cash"] == pytest.approx(1000.0 + notional)

        assert svc.get_position(bob["id"], mkt["id"])["yes_shares"] == pytest.approx(10.0)
        assert svc.get_position(alice["id"], mkt["id"])["yes_shares"] == pytest.approx(-10.0)

    def test_market_order(self, svc: MarketService):
        mkt = _make_open_cda(svc, slug="cda-mkt")
        seller = svc.create_agent(name="seller", cash=1000.0)
        buyer = svc.create_agent(name="buyer", cash=1000.0)

        svc.execute_cda_order(mkt["id"], seller["id"], "sell", 5.0, 0.60, "limit")
        result = svc.execute_cda_order(mkt["id"], buyer["id"], "buy", 5.0, None, "market")

        assert result["filled_quantity"] == pytest.approx(5.0)
        assert result["resting_order_id"] is None

    def test_partial_fill(self, svc: MarketService):
        mkt = _make_open_cda(svc, slug="cda-partial")
        seller = svc.create_agent(name="seller", cash=1000.0)
        buyer = svc.create_agent(name="buyer", cash=1000.0)

        svc.execute_cda_order(mkt["id"], seller["id"], "sell", 3.0, 0.50, "limit")
        result = svc.execute_cda_order(mkt["id"], buyer["id"], "buy", 5.0, 0.50, "limit")

        assert result["filled_quantity"] == pytest.approx(3.0)
        assert result["remaining_quantity"] == pytest.approx(2.0)
        assert result["resting_order_id"] is not None


# ── get_price_snapshot ─────────────────────────────────────────────────


class TestPriceSnapshot:
    def test_lmsr_snapshot(self, svc: MarketService):
        mkt = _make_open_lmsr(svc, slug="snap-lmsr")
        agent = svc.create_agent(name="alice", cash=10000.0)
        svc.execute_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)

        snap = svc.get_price_snapshot(mkt["id"])
        assert snap["mechanism"] == "lmsr"
        assert snap["price"] > 0.5
        assert snap["inv_yes"] > 0
        assert snap["b"] == 100.0
        assert snap["market_id"] == mkt["id"]
        assert snap["price"] == pytest.approx(svc.get_price(mkt["id"]))

    def test_cda_snapshot(self, svc: MarketService):
        mkt = _make_open_cda(svc, slug="snap-cda")
        alice = svc.create_agent(name="alice", cash=1000.0)
        bob = svc.create_agent(name="bob", cash=1000.0)
        svc.execute_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.60)
        svc.execute_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.40)

        snap = svc.get_price_snapshot(mkt["id"])
        assert snap["mechanism"] == "cda"
        assert snap["best_bid"] == pytest.approx(0.40)
        assert snap["best_ask"] == pytest.approx(0.60)
        assert snap["price"] == pytest.approx(0.50)

    def test_cda_snapshot_after_trade(self, svc: MarketService):
        mkt = _make_open_cda(svc, slug="snap-trade")
        alice = svc.create_agent(name="alice", cash=1000.0)
        bob = svc.create_agent(name="bob", cash=1000.0)
        svc.execute_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.55)
        svc.execute_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.55)

        snap = svc.get_price_snapshot(mkt["id"])
        assert snap["last_trade_price"] == pytest.approx(0.55)


# ── Concurrent LMSR trading ───────────────────────────────────────────


class TestConcurrentLMSR:
    def test_two_threads_consistent_state(self, svc: MarketService):
        mkt = _make_running_lmsr(svc, slug="race")
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
        expected_price = _lmsr_price(mkt_state["inv_yes"], mkt_state["inv_no"], mkt_state["b"])
        assert final_price == pytest.approx(expected_price)

        a1_pos = svc.get_position(a1["id"], mkt["id"])
        a2_pos = svc.get_position(a2["id"], mkt["id"])
        assert a1_pos["yes_shares"] == pytest.approx(trades_per_thread)
        assert a2_pos["yes_shares"] == pytest.approx(trades_per_thread)

    def test_opposing_trades_consistent(self, svc: MarketService):
        mkt = _make_running_lmsr(svc, slug="oppose")
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
        mkt = _make_running_lmsr(svc, slug="stress")
        n_threads = 8
        trades_each = 20
        agents = [svc.create_agent(name=f"agent-{i}", cash=100_000.0) for i in range(n_threads)]
        errors: list = []

        def worker(agent_id: int):
            try:
                for _ in range(trades_each):
                    svc.execute_trade(agent_id, mkt["id"], "buy_yes", shares=0.5)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(a["id"],)) for a in agents]
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
        def run_sequential():
            db = str(tmp_path / "seq.db")
            s = MarketService(db)
            mkt = s.create_market(slug="seq", title="Seq", mechanism="lmsr", b=100.0)
            s.set_market_status(mkt["id"], "running")
            a1 = s.create_agent(name="a1", cash=100_000.0)
            a2 = s.create_agent(name="a2", cash=100_000.0)
            for _ in range(30):
                s.execute_trade(a1["id"], mkt["id"], "buy_yes", shares=1.0)
            for _ in range(30):
                s.execute_trade(a2["id"], mkt["id"], "buy_yes", shares=1.0)
            result = {"price": s.get_price(mkt["id"]), "inv_yes": s.get_market(mkt["id"])["inv_yes"]}
            s.close()
            return result

        def run_concurrent():
            db = str(tmp_path / "conc.db")
            s = MarketService(db)
            mkt = s.create_market(slug="conc", title="Conc", mechanism="lmsr", b=100.0)
            s.set_market_status(mkt["id"], "running")
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
            result = {"price": s.get_price(mkt["id"]), "inv_yes": s.get_market(mkt["id"])["inv_yes"]}
            s.close()
            return result

        seq = run_sequential()
        conc = run_concurrent()
        assert seq["inv_yes"] == pytest.approx(conc["inv_yes"])
        assert seq["price"] == pytest.approx(conc["price"])


# ── Concurrent CDA trading ────────────────────────────────────────────


class TestConcurrentCDA:
    def test_two_threads_limit_orders(self, svc: MarketService):
        mkt = _make_open_cda(svc, slug="cda-race")
        seller = svc.create_agent(name="seller", cash=100_000.0)
        buyer = svc.create_agent(name="buyer", cash=100_000.0)

        n = 30
        errors: list = []

        def sell_worker():
            try:
                for _ in range(n):
                    svc.execute_limit_order(seller["id"], mkt["id"], "sell", quantity=1.0, price=0.50)
            except Exception as exc:
                errors.append(exc)

        def buy_worker():
            try:
                for _ in range(n):
                    svc.execute_limit_order(buyer["id"], mkt["id"], "buy", quantity=1.0, price=0.50)
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


# ── 10-thread stress tests ────────────────────────────────────────────


class TestTenThreadStress:
    def test_10_threads_lmsr_no_corruption(self, svc: MarketService):
        """10 threads each trade 20 times; final state is consistent."""
        mkt = _make_running_lmsr(svc, slug="stress10")
        n_threads = 10
        trades_each = 20
        agents = [svc.create_agent(name=f"stress-{i}", cash=100_000.0) for i in range(n_threads)]
        errors: list = []

        def worker(agent_id: int):
            try:
                for _ in range(trades_each):
                    svc.execute_trade(agent_id, mkt["id"], "buy_yes", shares=1.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(a["id"],)) for a in agents]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

        all_trades = svc.get_trades(market_id=mkt["id"], limit=500)
        assert len(all_trades) == n_threads * trades_each

        total_shares = sum(
            svc.get_position(a["id"], mkt["id"])["yes_shares"] for a in agents
        )
        assert total_shares == pytest.approx(n_threads * trades_each * 1.0)

        total_cost = sum(100_000.0 - svc.get_agent(a["id"])["cash"] for a in agents)
        assert total_cost > 0

    def test_10_threads_execute_lmsr_trade(self, svc: MarketService):
        """10 threads using the new execute_lmsr_trade API concurrently."""
        mkt = _make_running_lmsr(svc, slug="stress10-new")
        n_threads = 10
        trades_each = 20
        agents = [svc.create_agent(name=f"new-{i}", cash=100_000.0) for i in range(n_threads)]
        errors: list = []

        def worker(aid: int):
            try:
                for _ in range(trades_each):
                    svc.execute_lmsr_trade(mkt["id"], aid, quantity=1.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(a["id"],)) for a in agents]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

        all_trades = svc.get_trades(market_id=mkt["id"], limit=500)
        assert len(all_trades) == n_threads * trades_each

        trade_ids = [t["id"] for t in all_trades]
        assert len(trade_ids) == len(set(trade_ids)), "Duplicate trade IDs"

        mkt_state = svc.get_market(mkt["id"])
        assert mkt_state["inv_yes"] == pytest.approx(n_threads * trades_each)

        total_shares = sum(
            svc.get_position(a["id"], mkt["id"])["yes_shares"] for a in agents
        )
        assert total_shares == pytest.approx(n_threads * trades_each)
