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


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def store():
    s = MarketStore()
    yield s
    s.close()


# ── Market CRUD ────────────────────────────────────────────────────────


class TestMarketCRUD:
    def test_create_lmsr_market(self, store: MarketStore):
        mkt = store.create_market(slug="btc-100k", title="BTC > $100k?", mechanism="lmsr", b=100.0)
        assert mkt["id"] == 1
        assert mkt["slug"] == "btc-100k"
        assert mkt["mechanism"] == "lmsr"
        assert mkt["b"] == 100.0
        assert mkt["status"] == "open"

    def test_create_cda_market(self, store: MarketStore):
        mkt = store.create_market(
            slug="eth-10k", title="ETH > $10k?", mechanism="cda",
            tick_size=0.01, initial_price=0.5,
        )
        assert mkt["mechanism"] == "cda"
        assert mkt["tick_size"] == 0.01
        assert mkt["b"] is None

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
        assert len(store.list_markets(status="open")) == 2
        assert len(store.list_markets(status="resolved")) == 0

    def test_get_nonexistent_market_raises(self, store: MarketStore):
        with pytest.raises(ValueError, match="not found"):
            store.get_market(999)

    def test_lmsr_initial_price_is_fifty_fifty(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        assert store.get_price(mkt["id"]) == pytest.approx(0.5)

    def test_cda_initial_price_fallback(self, store: MarketStore):
        mkt = store.create_market(
            slug="test", title="Test", mechanism="cda", initial_price=0.6,
        )
        assert store.get_price(mkt["id"]) == pytest.approx(0.6)


# ── Agent CRUD ─────────────────────────────────────────────────────────


class TestAgentCRUD:
    def test_create_and_get(self, store: MarketStore):
        agent = store.create_agent(name="alice", cash=1000.0)
        assert agent["id"] == 1
        assert agent["name"] == "alice"
        assert agent["cash"] == 1000.0

        fetched = store.get_agent(agent["id"])
        assert fetched["name"] == "alice"

    def test_duplicate_name_rejected(self, store: MarketStore):
        store.create_agent(name="alice", cash=100.0)
        with pytest.raises(Exception):
            store.create_agent(name="alice", cash=200.0)

    def test_get_nonexistent_agent_raises(self, store: MarketStore):
        with pytest.raises(ValueError, match="not found"):
            store.get_agent(999)


# ── LMSR Trading ──────────────────────────────────────────────────────


class TestLMSRTrading:
    def test_buy_yes_moves_price_up(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        trade = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)

        assert trade["price_before"] == pytest.approx(0.5)
        assert trade["price_after"] > 0.5
        assert trade["cost"] > 0
        assert store.get_price(mkt["id"]) == pytest.approx(trade["price_after"])

    def test_sell_yes_moves_price_down(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        trade = store.submit_trade(agent["id"], mkt["id"], "sell_yes", shares=10.0)

        assert trade["price_after"] < 0.5
        assert trade["cost"] < 0
        assert store.get_price(mkt["id"]) == pytest.approx(trade["price_after"])

    def test_cash_decreases_after_buy(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        trade = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)

        alice = store.get_agent(agent["id"])
        assert alice["cash"] == pytest.approx(1000.0 - trade["cost"])

    def test_position_tracks_shares(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
        pos = store.get_position(agent["id"], mkt["id"])
        assert pos["yes_shares"] == pytest.approx(10.0)

        store.submit_trade(agent["id"], mkt["id"], "sell_yes", shares=3.0)
        pos = store.get_position(agent["id"], mkt["id"])
        assert pos["yes_shares"] == pytest.approx(7.0)

    def test_position_default_when_none(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)
        pos = store.get_position(agent["id"], mkt["id"])
        assert pos["yes_shares"] == 0.0

    def test_insufficient_funds_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="broke", cash=0.01)

        with pytest.raises(ValueError, match="Insufficient funds"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1000.0)

        assert store.get_price(mkt["id"]) == pytest.approx(0.5)
        assert store.get_agent(agent["id"])["cash"] == pytest.approx(0.01)

    def test_trade_on_closed_market_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)
        store.resolve_market(mkt["id"], "yes")

        with pytest.raises(ValueError, match="not open"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)

    def test_invalid_side_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        with pytest.raises(ValueError, match="side must be"):
            store.submit_trade(agent["id"], mkt["id"], "buy_no", shares=1.0)

    def test_zero_shares_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        with pytest.raises(ValueError, match="positive"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=0.0)

    def test_trade_log_recorded(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=5.0)
        store.submit_trade(agent["id"], mkt["id"], "sell_yes", shares=2.0)

        trades = store.get_trades(market_id=mkt["id"])
        assert len(trades) == 2
        assert trades[0]["side"] == "sell_yes"
        assert trades[1]["side"] == "buy_yes"

    def test_lmsr_trade_on_cda_market_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda")
        agent = store.create_agent(name="alice", cash=1000.0)

        with pytest.raises(ValueError, match="LMSR"):
            store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=1.0)


# ── CDA Trading ───────────────────────────────────────────────────────


class TestCDATrading:
    def test_limit_order_rests_on_empty_book(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        agent = store.create_agent(name="alice", cash=1000.0)

        result = store.submit_limit_order(agent["id"], mkt["id"], "buy", quantity=10.0, price=0.45)

        assert result["filled_quantity"] == pytest.approx(0.0)
        assert result["remaining_quantity"] == pytest.approx(10.0)
        assert result["resting_order_id"] is not None
        assert len(result["trades"]) == 0

    def test_crossing_limit_orders_match(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.50)
        result = store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.50)

        assert result["filled_quantity"] == pytest.approx(5.0)
        assert result["remaining_quantity"] == pytest.approx(0.0)
        assert len(result["trades"]) == 1
        assert result["trades"][0]["buyer_id"] == bob["id"]
        assert result["trades"][0]["seller_id"] == alice["id"]
        assert result["trades"][0]["price"] == pytest.approx(0.50)
        assert result["trades"][0]["quantity"] == pytest.approx(5.0)

    def test_partial_fill(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=3.0, price=0.50)
        result = store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.50)

        assert result["filled_quantity"] == pytest.approx(3.0)
        assert result["remaining_quantity"] == pytest.approx(2.0)
        assert result["resting_order_id"] is not None

    def test_market_order_fills_against_book(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.50)
        result = store.submit_market_order(bob["id"], mkt["id"], "buy", quantity=3.0)

        assert result["filled_quantity"] == pytest.approx(3.0)
        assert result["remaining_quantity"] == pytest.approx(0.0)
        assert result["resting_order_id"] is None

    def test_market_order_no_resting(self, store: MarketStore):
        """Market orders never rest on the book, even if unfilled."""
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        agent = store.create_agent(name="alice", cash=1000.0)

        result = store.submit_market_order(agent["id"], mkt["id"], "buy", quantity=5.0)

        assert result["filled_quantity"] == pytest.approx(0.0)
        assert result["resting_order_id"] is None

    def test_cda_cash_and_positions_update(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=10.0, price=0.40)
        store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=10.0, price=0.40)

        notional = 0.40 * 10.0

        alice_after = store.get_agent(alice["id"])
        bob_after = store.get_agent(bob["id"])
        assert bob_after["cash"] == pytest.approx(1000.0 - notional)
        assert alice_after["cash"] == pytest.approx(1000.0 + notional)

        alice_pos = store.get_position(alice["id"], mkt["id"])
        bob_pos = store.get_position(bob["id"], mkt["id"])
        assert alice_pos["yes_shares"] == pytest.approx(-10.0)
        assert bob_pos["yes_shares"] == pytest.approx(10.0)

    def test_cda_price_updates_after_trade(self, store: MarketStore):
        mkt = store.create_market(
            slug="test", title="Test", mechanism="cda",
            tick_size=0.01, initial_price=0.5,
        )
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.60)
        store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.60)

        assert store.get_price(mkt["id"]) == pytest.approx(0.60)

    def test_order_book_state(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=5.0, price=0.55)
        store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=3.0, price=0.45)

        book = store.get_order_book(mkt["id"])
        assert len(book["bids"]) == 1
        assert book["bids"][0]["price"] == pytest.approx(0.45)
        assert book["bids"][0]["quantity"] == pytest.approx(3.0)
        assert len(book["asks"]) == 1
        assert book["asks"][0]["price"] == pytest.approx(0.55)
        assert book["asks"][0]["quantity"] == pytest.approx(5.0)
        assert book["best_bid"] == pytest.approx(0.45)
        assert book["best_ask"] == pytest.approx(0.55)

    def test_price_time_priority(self, store: MarketStore):
        """First order at a price level fills first (FIFO)."""
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)
        charlie = store.create_agent(name="charlie", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=3.0, price=0.50)
        store.submit_limit_order(bob["id"], mkt["id"], "sell", quantity=3.0, price=0.50)

        result = store.submit_market_order(charlie["id"], mkt["id"], "buy", quantity=3.0)

        assert len(result["trades"]) == 1
        assert result["trades"][0]["seller_id"] == alice["id"]

    def test_cancel_agent_orders(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        agent = store.create_agent(name="alice", cash=1000.0)

        store.submit_limit_order(agent["id"], mkt["id"], "buy", quantity=5.0, price=0.45)
        store.submit_limit_order(agent["id"], mkt["id"], "sell", quantity=3.0, price=0.55)

        cancelled = store.cancel_agent_orders(agent["id"], mkt["id"])
        assert cancelled == 2

        book = store.get_order_book(mkt["id"])
        assert len(book["bids"]) == 0
        assert len(book["asks"]) == 0

    def test_cda_order_on_lmsr_market_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="alice", cash=1000.0)

        with pytest.raises(ValueError, match="CDA"):
            store.submit_limit_order(agent["id"], mkt["id"], "buy", quantity=1.0, price=0.5)

    def test_invalid_side_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda")
        agent = store.create_agent(name="alice", cash=1000.0)

        with pytest.raises(ValueError, match="side must be"):
            store.submit_limit_order(agent["id"], mkt["id"], "buy_yes", quantity=1.0, price=0.5)

    def test_sell_side_matching(self, store: MarketStore):
        """Sell market order matches against resting bids."""
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=5.0, price=0.45)
        result = store.submit_market_order(alice["id"], mkt["id"], "sell", quantity=3.0)

        assert result["filled_quantity"] == pytest.approx(3.0)
        assert result["trades"][0]["seller_id"] == alice["id"]
        assert result["trades"][0]["buyer_id"] == bob["id"]
        assert result["trades"][0]["price"] == pytest.approx(0.45)
        assert result["trades"][0]["aggressor_side"] == "sell"

    def test_multiple_price_levels(self, store: MarketStore):
        """Aggressor walks through multiple price levels."""
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)
        charlie = store.create_agent(name="charlie", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=2.0, price=0.50)
        store.submit_limit_order(bob["id"], mkt["id"], "sell", quantity=3.0, price=0.55)

        result = store.submit_market_order(charlie["id"], mkt["id"], "buy", quantity=4.0)

        assert result["filled_quantity"] == pytest.approx(4.0)
        assert len(result["trades"]) == 2
        assert result["trades"][0]["price"] == pytest.approx(0.50)
        assert result["trades"][0]["quantity"] == pytest.approx(2.0)
        assert result["trades"][1]["price"] == pytest.approx(0.55)
        assert result["trades"][1]["quantity"] == pytest.approx(2.0)


# ── Multi-market ───────────────────────────────────────────────────────


class TestMultiMarket:
    def test_independent_prices_lmsr(self, store: MarketStore):
        m1 = store.create_market(slug="event-a", title="Event A", mechanism="lmsr", b=100.0)
        m2 = store.create_market(slug="event-b", title="Event B", mechanism="lmsr", b=50.0)
        agent = store.create_agent(name="bob", cash=5000.0)

        store.submit_trade(agent["id"], m1["id"], "buy_yes", shares=5.0)
        store.submit_trade(agent["id"], m2["id"], "sell_yes", shares=3.0)

        assert store.get_price(m1["id"]) > 0.5
        assert store.get_price(m2["id"]) < 0.5

    def test_mixed_mechanism_markets(self, store: MarketStore):
        m_lmsr = store.create_market(slug="lmsr-mkt", title="LMSR", mechanism="lmsr", b=100.0)
        m_cda = store.create_market(slug="cda-mkt", title="CDA", mechanism="cda", tick_size=0.01)

        alice = store.create_agent(name="alice", cash=5000.0)
        bob = store.create_agent(name="bob", cash=5000.0)

        store.submit_trade(alice["id"], m_lmsr["id"], "buy_yes", shares=10.0)
        assert store.get_price(m_lmsr["id"]) > 0.5

        store.submit_limit_order(alice["id"], m_cda["id"], "sell", quantity=5.0, price=0.60)
        store.submit_limit_order(bob["id"], m_cda["id"], "buy", quantity=5.0, price=0.60)
        assert store.get_price(m_cda["id"]) == pytest.approx(0.60)

    def test_separate_positions(self, store: MarketStore):
        m1 = store.create_market(slug="event-a", title="Event A", mechanism="lmsr", b=100.0)
        m2 = store.create_market(slug="event-b", title="Event B", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="bob", cash=5000.0)

        store.submit_trade(agent["id"], m1["id"], "buy_yes", shares=10.0)
        store.submit_trade(agent["id"], m2["id"], "buy_yes", shares=20.0)

        assert store.get_position(agent["id"], m1["id"])["yes_shares"] == pytest.approx(10.0)
        assert store.get_position(agent["id"], m2["id"])["yes_shares"] == pytest.approx(20.0)

    def test_shared_cash_across_markets(self, store: MarketStore):
        m1 = store.create_market(slug="event-a", title="Event A", mechanism="lmsr", b=100.0)
        m2 = store.create_market(slug="event-b", title="Event B", mechanism="lmsr", b=100.0)
        agent = store.create_agent(name="bob", cash=1000.0)

        t1 = store.submit_trade(agent["id"], m1["id"], "buy_yes", shares=5.0)
        t2 = store.submit_trade(agent["id"], m2["id"], "buy_yes", shares=5.0)

        bob = store.get_agent(agent["id"])
        assert bob["cash"] == pytest.approx(1000.0 - t1["cost"] - t2["cost"])

    def test_trade_log_filters(self, store: MarketStore):
        m1 = store.create_market(slug="a", title="A", mechanism="lmsr", b=100.0)
        m2 = store.create_market(slug="b", title="B", mechanism="lmsr", b=100.0)
        a1 = store.create_agent(name="alice", cash=5000.0)
        a2 = store.create_agent(name="bob", cash=5000.0)

        store.submit_trade(a1["id"], m1["id"], "buy_yes", shares=5.0)
        store.submit_trade(a2["id"], m1["id"], "buy_yes", shares=3.0)
        store.submit_trade(a1["id"], m2["id"], "sell_yes", shares=2.0)

        assert len(store.get_trades(market_id=m1["id"])) == 2
        assert len(store.get_trades(market_id=m2["id"])) == 1
        assert len(store.get_trades(agent_id=a1["id"])) == 2
        assert len(store.get_trades(agent_id=a2["id"])) == 1


# ── Resolution / settlement ───────────────────────────────────────────


class TestResolution:
    def test_resolve_lmsr_yes_pays_holders(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        alice = store.create_agent(name="alice", cash=1000.0)

        store.submit_trade(alice["id"], mkt["id"], "buy_yes", shares=10.0)
        cash_after_trade = store.get_agent(alice["id"])["cash"]

        result = store.resolve_market(mkt["id"], "yes")
        assert result["outcome"] == "yes"
        assert result["positions_settled"] == 1

        alice_final = store.get_agent(alice["id"])
        assert alice_final["cash"] == pytest.approx(cash_after_trade + 10.0)

    def test_resolve_cda_cancels_open_orders(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="cda", tick_size=0.01)
        alice = store.create_agent(name="alice", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "buy", quantity=5.0, price=0.45)
        book_before = store.get_order_book(mkt["id"])
        assert len(book_before["bids"]) == 1

        store.resolve_market(mkt["id"], "yes")

        book_after = store.get_order_book(mkt["id"])
        assert len(book_after["bids"]) == 0

    def test_resolve_no_pays_nothing(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        alice = store.create_agent(name="alice", cash=1000.0)

        store.submit_trade(alice["id"], mkt["id"], "buy_yes", shares=10.0)
        cash_after_trade = store.get_agent(alice["id"])["cash"]

        store.resolve_market(mkt["id"], "no")

        alice_final = store.get_agent(alice["id"])
        assert alice_final["cash"] == pytest.approx(cash_after_trade)

    def test_double_resolve_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        store.resolve_market(mkt["id"], "yes")

        with pytest.raises(ValueError, match="already resolved"):
            store.resolve_market(mkt["id"], "no")

    def test_invalid_outcome_rejected(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        with pytest.raises(ValueError, match="outcome must be"):
            store.resolve_market(mkt["id"], "maybe")

    def test_market_status_after_resolve(self, store: MarketStore):
        mkt = store.create_market(slug="test", title="Test", mechanism="lmsr", b=100.0)
        store.resolve_market(mkt["id"], "no")
        fetched = store.get_market(mkt["id"])
        assert fetched["status"] == "resolved"
        assert fetched["resolution"] == "no"
        assert fetched["resolved_at"] is not None


# ── Full lifecycle ─────────────────────────────────────────────────────


class TestFullLifecycle:
    def test_lmsr_lifecycle(self, store: MarketStore):
        """Create LMSR market → agent → trade → read price."""
        mkt = store.create_market(slug="btc-100k", title="BTC > $100k?", mechanism="lmsr", b=100.0)
        assert store.get_price(mkt["id"]) == pytest.approx(0.5)

        agent = store.create_agent(name="alice", cash=1000.0)

        trade = store.submit_trade(agent["id"], mkt["id"], side="buy_yes", shares=10.0)
        assert trade["price_before"] == pytest.approx(0.5)
        assert trade["price_after"] > 0.5
        assert trade["cost"] > 0

        new_price = store.get_price(mkt["id"])
        assert new_price == pytest.approx(trade["price_after"])
        assert new_price > 0.5

        alice = store.get_agent(agent["id"])
        assert alice["cash"] == pytest.approx(1000.0 - trade["cost"])

        pos = store.get_position(agent["id"], mkt["id"])
        assert pos["yes_shares"] == pytest.approx(10.0)

        trades = store.get_trades(market_id=mkt["id"])
        assert len(trades) == 1

    def test_cda_lifecycle(self, store: MarketStore):
        """Create CDA market → agents → limit orders → match → read price."""
        mkt = store.create_market(
            slug="eth-10k", title="ETH > $10k?", mechanism="cda",
            tick_size=0.01, initial_price=0.5,
        )
        assert store.get_price(mkt["id"]) == pytest.approx(0.5)

        alice = store.create_agent(name="alice", cash=1000.0)
        bob = store.create_agent(name="bob", cash=1000.0)

        store.submit_limit_order(alice["id"], mkt["id"], "sell", quantity=10.0, price=0.55)
        result = store.submit_limit_order(bob["id"], mkt["id"], "buy", quantity=10.0, price=0.55)

        assert result["filled_quantity"] == pytest.approx(10.0)
        assert len(result["trades"]) == 1

        notional = 0.55 * 10.0
        assert store.get_agent(bob["id"])["cash"] == pytest.approx(1000.0 - notional)
        assert store.get_agent(alice["id"])["cash"] == pytest.approx(1000.0 + notional)

        assert store.get_position(bob["id"], mkt["id"])["yes_shares"] == pytest.approx(10.0)
        assert store.get_position(alice["id"], mkt["id"])["yes_shares"] == pytest.approx(-10.0)

        assert store.get_price(mkt["id"]) == pytest.approx(0.55)
