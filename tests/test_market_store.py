"""
Unit tests for MarketStore — SQLite-backed multi-market LMSR + CDA layer.

Every test uses an in-memory database, no SimulationEngine involvement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app"
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from market_store import MarketStore


@pytest.fixture
def store():
    s = MarketStore()
    yield s
    s.close()


def _make_open_lmsr(store, slug="test", b=100.0, **kw):
    """Helper: create an LMSR market and set it to 'open' so trading is allowed."""
    mkt = store.create_market(slug=slug, title=slug, mechanism="lmsr", b=b, **kw)
    store.set_market_status(mkt["id"], "open")
    return store.get_market(mkt["id"])


def _make_running_lmsr(store, slug="test", b=100.0, **kw):
    mkt = store.create_market(slug=slug, title=slug, mechanism="lmsr", b=b, **kw)
    store.set_market_status(mkt["id"], "running")
    return store.get_market(mkt["id"])


def _make_open_cda(store, slug="test", tick_size=0.01, **kw):
    mkt = store.create_market(slug=slug, title=slug, mechanism="cda", tick_size=tick_size, **kw)
    store.set_market_status(mkt["id"], "open")
    return store.get_market(mkt["id"])


# ── Market CRUD ────────────────────────────────────────────────────────


class TestMarketCRUD:
    def test_create_lmsr_market(self, store: MarketStore):
        mkt = store.create_market(
            slug="btc-100k", title="BTC > $100k?", mechanism="lmsr",
            b=100.0, ground_truth=0.7,
        )
        assert mkt["id"] == 1
        assert mkt["mechanism"] == "lmsr"
        assert mkt["ground_truth"] == 0.7
        assert mkt["b"] == 100.0
        assert mkt["status"] == "created"

    def test_create_cda_market(self, store: MarketStore):
        mkt = store.create_market(
            slug="eth-10k", title="ETH > $10k?", mechanism="cda",
            tick_size=0.01, initial_price=0.5,
        )
        assert mkt["mechanism"] == "cda"
        assert mkt["tick_size"] == 0.01
        assert mkt["b"] is None

    def test_ground_truth_optional(self, store: MarketStore):
        mkt = store.create_market(slug="x", title="X", mechanism="lmsr", b=50.0)
        assert mkt["ground_truth"] is None

    def test_invalid_mechanism_rejected(self, store: MarketStore):
        with pytest.raises(ValueError, match="mechanism must be"):
            store.create_market(slug="x", title="X", mechanism="amm")

    def test_lmsr_requires_b(self, store: MarketStore):
        with pytest.raises(ValueError, match="b.*required"):
            store.create_market(slug="x", title="X", mechanism="lmsr")

    def test_duplicate_slug_rejected(self, store: MarketStore):
        store.create_market(slug="dup", title="First", mechanism="lmsr", b=50.0)
        with pytest.raises(Exception):
            store.create_market(slug="dup", title="Second", mechanism="lmsr", b=50.0)

    def test_list_markets(self, store: MarketStore):
        store.create_market(slug="a", title="A", mechanism="lmsr", b=100.0)
        store.create_market(slug="b", title="B", mechanism="cda")
        assert len(store.list_markets()) == 2
        assert len(store.list_markets(status="created")) == 2

    def test_get_nonexistent_market_raises(self, store: MarketStore):
        with pytest.raises(ValueError, match="not found"):
            store.get_market(999)

    def test_lmsr_initial_price_is_fifty_fifty(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        assert store.get_price(mkt["id"]) == pytest.approx(0.5)

    def test_cda_initial_price_fallback(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="cda", initial_price=0.6)
        assert store.get_price(mkt["id"]) == pytest.approx(0.6)


# ── Market status lifecycle ────────────────────────────────────────────


class TestMarketStatus:
    def test_default_status_is_created(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        assert mkt["status"] == "created"

    def test_set_status_to_running(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        updated = store.set_market_status(mkt["id"], "running")
        assert updated["status"] == "running"

    def test_set_status_to_open(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        updated = store.set_market_status(mkt["id"], "open")
        assert updated["status"] == "open"

    def test_set_status_to_stopped(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        updated = store.set_market_status(mkt["id"], "stopped")
        assert updated["status"] == "stopped"

    def test_invalid_status_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        with pytest.raises(ValueError, match="status must be"):
            store.set_market_status(mkt["id"], "invalid")

    def test_resolved_market_cannot_change_status(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        store.resolve_market(mkt["id"], "yes")
        with pytest.raises(ValueError, match="resolved"):
            store.set_market_status(mkt["id"], "running")

    def test_cannot_trade_on_created_market(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)
        with pytest.raises(ValueError, match="trading requires"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)

    def test_cannot_trade_on_stopped_market(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        store.set_market_status(mkt["id"], "stopped")
        agent = store.create_agent(name="alice", cash=1000.0)
        with pytest.raises(ValueError, match="trading requires"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)

    def test_can_trade_on_running_market(self, store: MarketStore):
        mkt = _make_running_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        trade = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)
        assert trade["trade_id"] is not None


# ── Agent CRUD ─────────────────────────────────────────────────────────


class TestAgentCRUD:
    def test_create_basic_agent(self, store: MarketStore):
        agent = store.create_agent(name="alice", cash=1000.0)
        assert agent["id"] == 1
        assert agent["name"] == "alice"
        assert agent["cash"] == 1000.0

    def test_create_agent_with_market_position_created_lazily(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = store.create_agent(
            name="alice", cash=1000.0,
            belief=0.6, rho=1.5, personality="aggressive",
        )
        assert agent["belief"] == 0.6
        assert agent["rho"] == 1.5
        assert agent["personality"] == "aggressive"

        store.ensure_position(agent["id"], mkt["id"])
        pos = store.get_position(agent["id"], mkt["id"])
        assert pos["belief"] is not None
        assert 0.01 <= float(pos["belief"]) <= 0.99
        assert pos["rho"] == 1.5
        assert pos["personality"] == "aggressive"
        assert pos["yes_shares"] == 0.0

    def test_get_agent_is_global(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = store.create_agent(
            name="bob", cash=500.0,
            belief=0.3, rho=2.0, personality="cautious",
        )
        fetched = store.get_agent(agent["id"])
        assert fetched["cash"] == 500.0
        assert fetched["belief"] == 0.3
        assert fetched["rho"] == 2.0
        assert fetched["personality"] == "cautious"
        store.ensure_position(agent["id"], mkt["id"])
        assert store.get_position(agent["id"], mkt["id"])["yes_shares"] == pytest.approx(0.0)

    def test_get_agent_without_market(self, store: MarketStore):
        agent = store.create_agent(name="alice", cash=1000.0)
        fetched = store.get_agent(agent["id"])
        assert "belief" in fetched
        assert fetched["belief"] is None
        assert fetched["rho"] is None
        assert fetched["personality"] is None

    def test_duplicate_name_rejected(self, store: MarketStore):
        store.create_agent(name="alice", cash=100.0)
        with pytest.raises(Exception):
            store.create_agent(name="alice", cash=200.0)

    def test_get_nonexistent_agent_raises(self, store: MarketStore):
        with pytest.raises(ValueError, match="not found"):
            store.get_agent(999)


# ── Agent belief ───────────────────────────────────────────────────────


class TestAgentBelief:
    def test_set_belief(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = store.create_agent(
            name="alice", cash=1000.0, belief=0.5,
        )
        old = store.set_agent_belief(agent["id"], mkt["id"], 0.8)
        assert old is not None
        assert 0.01 <= float(old) <= 0.99

        store.ensure_position(agent["id"], mkt["id"])
        pos = store.get_position(agent["id"], mkt["id"])
        assert pos["belief"] == 0.8

    def test_set_belief_only_updates_target_market(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        mkt2 = store.create_market(slug="u", title="U", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)
        store.ensure_position(agent["id"], mkt["id"])
        store.ensure_position(agent["id"], mkt2["id"])
        old = store.set_agent_belief(agent["id"], mkt["id"], 0.8)
        assert old is not None
        assert store.get_position(agent["id"], mkt["id"])["belief"] == pytest.approx(0.8)
        assert store.get_position(agent["id"], mkt2["id"])["belief"] != pytest.approx(0.8)


class TestPositionLinking:
    def test_ensure_position_creates_row(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0, belief=0.6, rho=1.2)
        pos0 = store.get_position(agent["id"], mkt["id"])
        assert pos0["yes_shares"] == pytest.approx(0.0)

        store.ensure_position(agent["id"], mkt["id"])
        pos1 = store.get_position(agent["id"], mkt["id"])
        assert pos1["yes_shares"] == pytest.approx(0.0)
        assert pos1["belief"] is not None
        assert 0.01 <= float(pos1["belief"]) <= 0.99
        assert pos1["rho"] == pytest.approx(1.2)

    def test_initial_market_beliefs_are_agent_specific(self, store: MarketStore):
        mkt = store.create_market(
            slug="noise",
            title="Noise",
            mechanism="lmsr",
            b=100.0,
            ground_truth=0.7,
        )
        a1 = store.create_agent(name="n-a1", cash=1000.0, belief=0.7)
        a2 = store.create_agent(name="n-a2", cash=1000.0, belief=0.7)
        p1 = store.ensure_position(a1["id"], mkt["id"])
        p2 = store.ensure_position(a2["id"], mkt["id"])
        assert p1["belief"] is not None and p2["belief"] is not None
        assert p1["belief"] != pytest.approx(p2["belief"])


# ── Agent portfolio update ─────────────────────────────────────────────


class TestAgentPortfolio:
    def test_update_portfolio(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        result = store.update_agent_portfolio(mkt["id"], agent["id"], cash_delta=-50.0, shares_delta=10.0)
        assert result["cash"] == pytest.approx(950.0)
        assert result["yes_shares"] == pytest.approx(10.0)

    def test_update_portfolio_accumulates(self, store: MarketStore):
        mkt = store.create_market(slug="t", title="T", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        store.update_agent_portfolio(mkt["id"], agent["id"], cash_delta=-50.0, shares_delta=10.0)
        store.update_agent_portfolio(mkt["id"], agent["id"], cash_delta=-30.0, shares_delta=5.0)

        pos = store.get_position(agent["id"], mkt["id"])
        assert pos["yes_shares"] == pytest.approx(15.0)
        assert store.get_agent(agent["id"])["cash"] == pytest.approx(920.0)


# ── LMSR Trading ──────────────────────────────────────────────────────


class TestLMSRTrading:
    def test_buy_yes_moves_price_up(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        trade = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
        assert trade["price_before"] == pytest.approx(0.5)
        assert trade["price_after"] > 0.5
        assert store.get_price(mkt["id"]) == pytest.approx(trade["price_after"])

    def test_sell_yes_moves_price_down(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        trade = store.submit_trade(agent["id"], mkt["id"], "sell_yes", shares=10.0)
        assert trade["price_after"] < 0.5
        assert trade["cost"] < 0

    def test_cash_decreases_after_buy(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        trade = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
        assert store.get_agent(agent["id"])["cash"] == pytest.approx(1000.0 - trade["cost"])

    def test_position_tracks_shares(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
        assert store.get_position(agent["id"], mkt["id"])["yes_shares"] == pytest.approx(10.0)
        store.submit_trade(agent["id"], mkt["id"], "sell_yes", shares=3.0)
        assert store.get_position(agent["id"], mkt["id"])["yes_shares"] == pytest.approx(7.0)

    def test_insufficient_funds_rejected(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="broke", cash=0.01)
        with pytest.raises(ValueError, match="Insufficient funds"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1000.0)
        assert store.get_price(mkt["id"]) == pytest.approx(0.5)

    def test_trade_on_resolved_market_rejected(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        store.resolve_market(mkt["id"], "yes")
        with pytest.raises(ValueError, match="trading requires"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)

    def test_trade_log_recorded(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=5.0)
        store.submit_trade(agent["id"], mkt["id"], "sell_yes", shares=2.0)
        trades = store.get_trades(market_id=mkt["id"])
        assert len(trades) == 2

    def test_lmsr_trade_on_cda_market_rejected(self, store: MarketStore):
        mkt = _make_open_cda(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        with pytest.raises(ValueError, match="LMSR"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)


# ── CDA Trading ───────────────────────────────────────────────────────


class TestCDATrading:
    def test_limit_order_rests_on_empty_book(self, store: MarketStore):
        mkt = _make_open_cda(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        result = store.submit_limit_order(agent["id"], mkt["id"], "buy", quantity=10.0, price=0.45)
        assert result["filled_quantity"] == pytest.approx(0.0)
        assert result["resting_order_id"] is not None

    def test_crossing_limit_orders_match(self, store: MarketStore):
        mkt = _make_open_cda(store)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)
        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.50)
        result = store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.50)
        assert result["filled_quantity"] == pytest.approx(5.0)
        assert len(result["trades"]) == 1

    def test_cda_order_on_lmsr_market_rejected(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=1000.0)
        with pytest.raises(ValueError, match="CDA"):
            store.submit_limit_order(agent["id"], mkt["id"], "buy", quantity=1.0, price=0.5)

    def test_price_time_priority(self, store: MarketStore):
        mkt = _make_open_cda(store)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)
        charlie = store.create_agent(name="charlie", cash=1000.0)
        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=3.0, price=0.50)
        store.submit_limit_order(bob["id"], mkt["id"], "sell", quantity=3.0, price=0.50)
        result = store.submit_market_order(charlie["id"], mkt["id"], "buy", quantity=3.0)
        assert result["trades"][0]["seller_id"] == alice["id"]


# ── Trade history with since_trade_id ──────────────────────────────────


class TestTradeHistory:
    def test_since_trade_id(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        agent = store.create_agent(name="alice", cash=10000.0)
        t1 = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)
        t2 = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)
        t3 = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)

        trades = store.get_trades(market_id=mkt["id"], since_trade_id=t1["trade_id"])
        assert len(trades) == 2
        ids = {t["id"] for t in trades}
        assert t1["trade_id"] not in ids
        assert t2["trade_id"] in ids
        assert t3["trade_id"] in ids


# ── Multi-market ───────────────────────────────────────────────────────


class TestMultiMarket:
    def test_independent_prices_lmsr(self, store: MarketStore):
        m1 = _make_open_lmsr(store, slug="a")
        m2 = _make_open_lmsr(store, slug="b", b=50.0)
        agent = store.create_agent(name="bob", cash=5000.0)
        store.submit_trade(agent["id"], m1["id"], "buy_yes", shares=5.0)
        store.submit_trade(agent["id"], m2["id"], "sell_yes", shares=3.0)
        assert store.get_price(m1["id"]) > 0.5
        assert store.get_price(m2["id"]) < 0.5

    def test_mixed_mechanism_markets(self, store: MarketStore):
        m_lmsr = _make_open_lmsr(store, slug="lmsr-mkt")
        m_cda = _make_open_cda(store, slug="cda-mkt")
        alice = store.create_agent(name="alice", cash=5000.0)
        bob = store.create_agent(name="bob", cash=5000.0)
        store.submit_trade(alice["id"], m_lmsr["id"], "buy_yes", shares=10.0)
        assert store.get_price(m_lmsr["id"]) > 0.5
        store.submit_limit_order(alice["id"], m_cda["id"], "sell", quantity=5.0, price=0.60)
        store.submit_limit_order(bob["id"], m_cda["id"], "buy", quantity=5.0, price=0.60)
        assert store.get_price(m_cda["id"]) == pytest.approx(0.60)

    def test_shared_cash_across_markets(self, store: MarketStore):
        m1 = _make_open_lmsr(store, slug="a")
        m2 = _make_open_lmsr(store, slug="b")
        agent = store.create_agent(name="bob", cash=1000.0)
        t1 = store.submit_trade(agent["id"], m1["id"], "buy_yes", shares=5.0)
        t2 = store.submit_trade(agent["id"], m2["id"], "buy_yes", shares=5.0)
        assert store.get_agent(agent["id"])["cash"] == pytest.approx(1000.0 - t1["cost"] - t2["cost"])


# ── Resolution ─────────────────────────────────────────────────────────


class TestResolution:
    def test_resolve_yes_pays_holders(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        alice = store.create_agent(name="alice", cash=1000.0)
        store.submit_trade(alice["id"], mkt["id"], "buy_yes", shares=10.0)
        cash_after = store.get_agent(alice["id"])["cash"]
        store.resolve_market(mkt["id"], "yes")
        assert store.get_agent(alice["id"])["cash"] == pytest.approx(cash_after + 10.0)

    def test_resolve_no_pays_nothing(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        alice = store.create_agent(name="alice", cash=1000.0)
        store.submit_trade(alice["id"], mkt["id"], "buy_yes", shares=10.0)
        cash_after = store.get_agent(alice["id"])["cash"]
        store.resolve_market(mkt["id"], "no")
        assert store.get_agent(alice["id"])["cash"] == pytest.approx(cash_after)

    def test_double_resolve_rejected(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        store.resolve_market(mkt["id"], "yes")
        with pytest.raises(ValueError, match="already resolved"):
            store.resolve_market(mkt["id"], "no")

    def test_market_status_after_resolve(self, store: MarketStore):
        mkt = _make_open_lmsr(store)
        store.resolve_market(mkt["id"], "no")
        fetched = store.get_market(mkt["id"])
        assert fetched["status"] == "resolved"
        assert fetched["resolution"] == "no"
        assert fetched["resolved_at"] is not None


# ── Full lifecycle ─────────────────────────────────────────────────────


class TestFullLifecycle:
    def test_lmsr_lifecycle(self, store: MarketStore):
        mkt = store.create_market(
            slug="btc", title="BTC > $100k?", mechanism="lmsr",
            b=100.0, ground_truth=0.7,
        )
        assert mkt["status"] == "created"
        assert mkt["ground_truth"] == 0.7

        store.set_market_status(mkt["id"], "running")

        agent = store.create_agent(
            name="alice", cash=1000.0,
            belief=0.6, rho=1.0, personality="moderate",
        )

        trade = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
        assert trade["price_after"] > 0.5

        old_belief = store.set_agent_belief(agent["id"], mkt["id"], 0.9)
        assert old_belief is not None
        assert 0.01 <= float(old_belief) <= 0.99

        agent_state = store.get_agent(agent["id"])
        assert agent_state["belief"] == 0.6
        assert store.get_position(agent["id"], mkt["id"])["belief"] == pytest.approx(0.9)
        assert store.get_position(agent["id"], mkt["id"])["yes_shares"] == pytest.approx(10.0)

    def test_cda_lifecycle(self, store: MarketStore):
        mkt = store.create_market(
            slug="eth", title="ETH > $10k?", mechanism="cda",
            tick_size=0.01, initial_price=0.5, ground_truth=0.4,
        )
        store.set_market_status(mkt["id"], "open")

        alice = store.create_agent(
            name="alice", cash=1000.0,
            belief=0.3, rho=2.0, personality="cautious",
        )
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=10.0, price=0.55)
        result = store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=10.0, price=0.55)
        assert result["filled_quantity"] == pytest.approx(10.0)
        assert store.get_price(mkt["id"]) == pytest.approx(0.55)
