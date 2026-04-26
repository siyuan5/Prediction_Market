"""
SQLite-backed persistent multi-market prediction market supporting LMSR and CDA.

Standalone state layer — no SimulationEngine dependency.  Every mutation
(trade, resolution) runs inside a single SQLite transaction so inventory,
cash, positions, and the trade log are always consistent.

LMSR markets trade against an automated market maker (cost-function pricing).
CDA markets use a full limit-order book with price-time priority matching.

Usage:
    store = MarketStore()                         # in-memory
    store = MarketStore("markets.db")             # file-backed

    mkt   = store.create_market("btc-100k", "BTC > $100k?",
                                mechanism="lmsr", b=200.0, ground_truth=0.7)
    agent = store.create_agent(
        "alice",
        cash=1000.0,
        belief=0.6,
        rho=1.0,
        personality="aggressive",
    )
    store.set_market_status(mkt["id"], "running")
    trade = store.submit_trade(agent["id"], mkt["id"], "buy_yes", shares=10.0)
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_VALID_MARKET_STATUSES = {"created", "open", "running", "stopped"}
_TRADEABLE_STATUSES = {"open", "running"}

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS markets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    slug             TEXT    UNIQUE NOT NULL,
    title            TEXT    NOT NULL,
    description      TEXT    NOT NULL DEFAULT '',
    mechanism        TEXT    NOT NULL DEFAULT 'lmsr',
    ground_truth     REAL,
    b                REAL,
    inv_yes          REAL    NOT NULL DEFAULT 0.0,
    inv_no           REAL    NOT NULL DEFAULT 0.0,
    tick_size        REAL,
    min_price        REAL    NOT NULL DEFAULT 0.001,
    max_price        REAL    NOT NULL DEFAULT 0.999,
    last_trade_price REAL,
    initial_price    REAL    NOT NULL DEFAULT 0.5,
    status           TEXT    NOT NULL DEFAULT 'created',
    resolution       TEXT,
    created_at       TEXT    NOT NULL,
    resolved_at      TEXT
);

CREATE TABLE IF NOT EXISTS agents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE NOT NULL,
    cash        REAL    NOT NULL,
    belief      REAL,
    rho         REAL,
    personality TEXT,
    created_at  TEXT    NOT NULL,
    deleted_at  TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   INTEGER NOT NULL REFERENCES agents(id),
    market_id  INTEGER NOT NULL REFERENCES markets(id),
    yes_shares REAL    NOT NULL DEFAULT 0.0,
    UNIQUE(agent_id, market_id)
);

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id    INTEGER NOT NULL REFERENCES markets(id),
    agent_id     INTEGER NOT NULL REFERENCES agents(id),
    side         TEXT    NOT NULL,
    shares       REAL    NOT NULL,
    cost         REAL    NOT NULL,
    price_before REAL    NOT NULL,
    price_after  REAL    NOT NULL,
    created_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id  INTEGER NOT NULL REFERENCES markets(id),
    agent_id   INTEGER NOT NULL REFERENCES agents(id),
    side       TEXT    NOT NULL,
    price      REAL    NOT NULL,
    quantity   REAL    NOT NULL,
    remaining  REAL    NOT NULL,
    status     TEXT    NOT NULL DEFAULT 'open',
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS news_events (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id              INTEGER NOT NULL REFERENCES markets(id),
    headline               TEXT    NOT NULL,
    mode                   TEXT    NOT NULL,
    requested_new_belief   REAL,
    requested_delta        REAL,
    affected_fraction      REAL    NOT NULL,
    min_signal_sensitivity REAL    NOT NULL,
    n_candidates           INTEGER NOT NULL,
    n_affected             INTEGER NOT NULL,
    mean_belief_before     REAL    NOT NULL,
    mean_belief_after      REAL    NOT NULL,
    created_at             TEXT    NOT NULL
);
"""


class MarketStore:
    """SQLite-backed multi-market prediction market (LMSR + CDA).

    When ``_conn`` is provided, the store wraps that connection instead of
    creating its own.  When ``_external_transactions`` is True, all internal
    transaction management is skipped (the caller — typically MarketService —
    is responsible for BEGIN / COMMIT / ROLLBACK).
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        *,
        _conn: Optional[sqlite3.Connection] = None,
        _external_transactions: bool = False,
    ):
        self._ext_txn = _external_transactions
        if _conn is not None:
            self.conn = _conn
            self._owns_conn = False
        else:
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(_SCHEMA)
            self._owns_conn = True
        self._migrate_agents_schema()

    def _migrate_agents_schema(self) -> None:
        """
        Ensure the agents table has global profile columns.

        Older databases may only have (id, name, cash, created_at). We keep this
        migration local and idempotent so Task 1 can run against existing files.
        """
        cols = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(agents)").fetchall()
        }
        if "belief" not in cols:
            self.conn.execute("ALTER TABLE agents ADD COLUMN belief REAL")
        if "rho" not in cols:
            self.conn.execute("ALTER TABLE agents ADD COLUMN rho REAL")
        if "personality" not in cols:
            self.conn.execute("ALTER TABLE agents ADD COLUMN personality TEXT")
        if "deleted_at" not in cols:
            self.conn.execute("ALTER TABLE agents ADD COLUMN deleted_at TEXT")

    def close(self) -> None:
        if self._owns_conn:
            self.conn.close()

    @contextmanager
    def _transaction(self):
        """Wraps the body in a transaction, unless externally managed."""
        if self._ext_txn:
            yield
        else:
            with self.conn:
                yield

    # ── LMSR math (pure functions) ─────────────────────────────────────

    @staticmethod
    def _lmsr_cost(inv_yes: float, inv_no: float, b: float) -> float:
        return b * math.log(math.exp(inv_yes / b) + math.exp(inv_no / b))

    @staticmethod
    def _lmsr_price(inv_yes: float, inv_no: float, b: float) -> float:
        e_yes = math.exp(inv_yes / b)
        e_no = math.exp(inv_no / b)
        return e_yes / (e_yes + e_no)

    # ── CDA helpers ────────────────────────────────────────────────────

    def _cda_best_bid(self, market_id: int) -> Optional[float]:
        row = self.conn.execute(
            "SELECT MAX(price) AS p FROM orders "
            "WHERE market_id = ? AND side = 'buy' AND status = 'open'",
            (market_id,),
        ).fetchone()
        return row["p"] if row and row["p"] is not None else None

    def _cda_best_ask(self, market_id: int) -> Optional[float]:
        row = self.conn.execute(
            "SELECT MIN(price) AS p FROM orders "
            "WHERE market_id = ? AND side = 'sell' AND status = 'open'",
            (market_id,),
        ).fetchone()
        return row["p"] if row and row["p"] is not None else None

    def _cda_reference_price(self, market_id: int) -> float:
        bid = self._cda_best_bid(market_id)
        ask = self._cda_best_ask(market_id)
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        mkt = self.conn.execute(
            "SELECT last_trade_price, initial_price FROM markets WHERE id = ?",
            (market_id,),
        ).fetchone()
        if mkt["last_trade_price"] is not None:
            return mkt["last_trade_price"]
        if bid is not None:
            return bid
        if ask is not None:
            return ask
        return mkt["initial_price"]

    def _normalize_price(self, price: float, tick_size: float,
                         min_price: float, max_price: float) -> float:
        clipped = min(max_price, max(min_price, float(price)))
        ticks = round(clipped / tick_size)
        return ticks * tick_size

    def _check_tradeable(self, mkt: sqlite3.Row) -> None:
        if mkt["status"] not in _TRADEABLE_STATUSES:
            raise ValueError(
                f"Market {mkt['id']} has status {mkt['status']!r}; "
                f"trading requires one of {_TRADEABLE_STATUSES}"
            )

    # ── Markets ────────────────────────────────────────────────────────

    def create_market(
        self,
        slug: str,
        title: str,
        *,
        mechanism: str = "lmsr",
        b: Optional[float] = None,
        ground_truth: Optional[float] = None,
        description: str = "",
        tick_size: float = 0.0001,
        min_price: float = 0.001,
        max_price: float = 0.999,
        initial_price: float = 0.5,
    ) -> Dict[str, Any]:
        if mechanism not in ("lmsr", "cda"):
            raise ValueError(f"mechanism must be 'lmsr' or 'cda', got {mechanism!r}")
        if mechanism == "lmsr" and b is None:
            raise ValueError("b (liquidity parameter) is required for LMSR markets")
        if mechanism == "lmsr" and b is not None and b <= 0:
            raise ValueError("b must be positive")

        now = datetime.now(timezone.utc).isoformat()
        with self._transaction():
            cur = self.conn.execute(
                "INSERT INTO markets "
                "(slug, title, description, mechanism, ground_truth, b, tick_size, "
                " min_price, max_price, initial_price, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (slug, title, description, mechanism, ground_truth, b, tick_size,
                 min_price, max_price, initial_price, now),
            )
            row = self.conn.execute(
                "SELECT * FROM markets WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        return self._market_row_to_dict(row)

    def get_market(self, market_id: int) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM markets WHERE id = ?", (market_id,)).fetchone()
        if row is None:
            raise ValueError(f"Market {market_id} not found")
        return self._market_row_to_dict(row)

    def list_markets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status is not None:
            rows = self.conn.execute(
                "SELECT * FROM markets WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM markets ORDER BY id").fetchall()
        return [self._market_row_to_dict(r) for r in rows]

    def get_price(self, market_id: int) -> float:
        row = self.conn.execute(
            "SELECT mechanism, inv_yes, inv_no, b FROM markets WHERE id = ?",
            (market_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Market {market_id} not found")
        if row["mechanism"] == "lmsr":
            return self._lmsr_price(row["inv_yes"], row["inv_no"], row["b"])
        return self._cda_reference_price(market_id)

    def set_market_status(self, market_id: int, status: str) -> Dict[str, Any]:
        if status not in _VALID_MARKET_STATUSES:
            raise ValueError(
                f"status must be one of {_VALID_MARKET_STATUSES}, got {status!r}"
            )
        with self._transaction():
            mkt = self.conn.execute(
                "SELECT * FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            if mkt["status"] == "resolved":
                raise ValueError(f"Market {market_id} is resolved and cannot change status")
            self.conn.execute(
                "UPDATE markets SET status = ? WHERE id = ?", (status, market_id)
            )
        return self.get_market(market_id)

    def get_order_book(self, market_id: int) -> Dict[str, Any]:
        bids = self.conn.execute(
            "SELECT price, SUM(remaining) as total_qty FROM orders "
            "WHERE market_id = ? AND side = 'buy' AND status = 'open' "
            "GROUP BY price ORDER BY price DESC",
            (market_id,),
        ).fetchall()
        asks = self.conn.execute(
            "SELECT price, SUM(remaining) as total_qty FROM orders "
            "WHERE market_id = ? AND side = 'sell' AND status = 'open' "
            "GROUP BY price ORDER BY price ASC",
            (market_id,),
        ).fetchall()
        return {
            "bids": [{"price": r["price"], "quantity": r["total_qty"]} for r in bids],
            "asks": [{"price": r["price"], "quantity": r["total_qty"]} for r in asks],
            "best_bid": self._cda_best_bid(market_id),
            "best_ask": self._cda_best_ask(market_id),
        }

    def resolve_market(self, market_id: int, outcome: str) -> Dict[str, Any]:
        if outcome not in ("yes", "no"):
            raise ValueError(f"outcome must be 'yes' or 'no', got {outcome!r}")
        payoff = 1.0 if outcome == "yes" else 0.0
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction():
            mkt = self.conn.execute("SELECT * FROM markets WHERE id = ?", (market_id,)).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            if mkt["status"] == "resolved":
                raise ValueError(f"Market {market_id} is already resolved")
            self.conn.execute(
                "UPDATE orders SET status = 'cancelled' "
                "WHERE market_id = ? AND status = 'open'",
                (market_id,),
            )
            self.conn.execute(
                "UPDATE markets SET status = 'resolved', resolution = ?, resolved_at = ? "
                "WHERE id = ?",
                (outcome, now, market_id),
            )
            positions = self.conn.execute(
                "SELECT agent_id, yes_shares FROM positions WHERE market_id = ?",
                (market_id,),
            ).fetchall()
            for pos in positions:
                payout = pos["yes_shares"] * payoff
                self.conn.execute(
                    "UPDATE agents SET cash = cash + ? WHERE id = ?",
                    (payout, pos["agent_id"]),
                )
        return {
            "market_id": market_id,
            "outcome": outcome,
            "positions_settled": len(positions),
        }

    # ── Agents ─────────────────────────────────────────────────────────

    def create_agent(
        self,
        name: str,
        cash: float,
        *,
        belief: Optional[float] = None,
        rho: Optional[float] = None,
        personality: Optional[str] = None,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction():
            cur = self.conn.execute(
                "INSERT INTO agents (name, cash, belief, rho, personality, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (name, cash, belief, rho, personality, now),
            )
            agent_id = cur.lastrowid
        row = self.conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return self._agent_row_to_dict(row)

    def get_agent(self, agent_id: int) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM agents WHERE id = ? AND deleted_at IS NULL",
            (agent_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Agent {agent_id} not found")
        return self._agent_row_to_dict(row)

    def list_agents(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        total_row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM agents WHERE deleted_at IS NULL"
        ).fetchone()
        total = int(total_row["c"] if total_row is not None else 0)
        rows = self.conn.execute(
            "SELECT * FROM agents WHERE deleted_at IS NULL ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "agents": [self._agent_row_to_dict(r) for r in rows],
            "total": total,
        }

    def update_agent(
        self,
        agent_id: int,
        *,
        name: Optional[str] = None,
        cash: Optional[float] = None,
        belief: Optional[float] = None,
        rho: Optional[float] = None,
        personality: Optional[str] = None,
    ) -> Dict[str, Any]:
        updates: List[str] = []
        params: List[Any] = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if cash is not None:
            updates.append("cash = ?")
            params.append(cash)
        if belief is not None:
            updates.append("belief = ?")
            params.append(belief)
        if rho is not None:
            updates.append("rho = ?")
            params.append(rho)
        if personality is not None:
            updates.append("personality = ?")
            params.append(personality)
        if not updates:
            return self.get_agent(agent_id)
        with self._transaction():
            exists = self.conn.execute(
                "SELECT id FROM agents WHERE id = ? AND deleted_at IS NULL",
                (agent_id,),
            ).fetchone()
            if exists is None:
                raise ValueError(f"Agent {agent_id} not found")
            params.append(agent_id)
            self.conn.execute(
                f"UPDATE agents SET {', '.join(updates)} WHERE id = ?",
                params,
            )
        return self.get_agent(agent_id)

    def set_agent_belief(self, agent_id: int, new_belief: float) -> float:
        """Update global agent belief and return the previous value."""
        with self._transaction():
            row = self.conn.execute(
                "SELECT belief FROM agents WHERE id = ? AND deleted_at IS NULL",
                (agent_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Agent {agent_id} not found")
            old_belief = row["belief"]
            self.conn.execute(
                "UPDATE agents SET belief = ? WHERE id = ?",
                (new_belief, agent_id),
            )
        return old_belief

    def soft_delete_agent(self, agent_id: int) -> Dict[str, Any]:
        """Mark an agent deleted without removing historical trades."""
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction():
            row = self.conn.execute(
                "SELECT * FROM agents WHERE id = ? AND deleted_at IS NULL",
                (agent_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Agent {agent_id} not found")
            self.conn.execute(
                "UPDATE agents SET deleted_at = ? WHERE id = ?",
                (now, agent_id),
            )
        deleted = dict(row)
        deleted["deleted_at"] = now
        return deleted

    def update_agent_portfolio(
        self,
        market_id: int,
        agent_id: int,
        cash_delta: float,
        shares_delta: float,
    ) -> Dict[str, Any]:
        """Directly adjust an agent's cash and shares for a market."""
        with self._transaction():
            agent = self.conn.execute(
                "SELECT * FROM agents WHERE id = ? AND deleted_at IS NULL", (agent_id,)
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")
            self.conn.execute(
                "UPDATE agents SET cash = cash + ? WHERE id = ?",
                (cash_delta, agent_id),
            )
            self.conn.execute(
                "INSERT INTO positions (agent_id, market_id, yes_shares) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(agent_id, market_id) "
                "DO UPDATE SET yes_shares = yes_shares + ?",
                (agent_id, market_id, shares_delta, shares_delta),
            )
        updated = self.get_agent(agent_id)
        pos = self.get_position(agent_id, market_id)
        updated["yes_shares"] = pos["yes_shares"]
        return updated

    # ── Positions ──────────────────────────────────────────────────────

    def ensure_position(self, agent_id: int, market_id: int) -> Dict[str, Any]:
        """Lazy-create an agent/market position row if it doesn't exist."""
        with self._transaction():
            agent = self.conn.execute(
                "SELECT id FROM agents WHERE id = ? AND deleted_at IS NULL",
                (agent_id,),
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")
            market = self.conn.execute(
                "SELECT id FROM markets WHERE id = ?",
                (market_id,),
            ).fetchone()
            if market is None:
                raise ValueError(f"Market {market_id} not found")
            self.conn.execute(
                "INSERT INTO positions (agent_id, market_id, yes_shares) "
                "VALUES (?, ?, 0.0) "
                "ON CONFLICT(agent_id, market_id) DO NOTHING",
                (agent_id, market_id),
            )
        return self.get_position(agent_id, market_id)

    def get_position(self, agent_id: int, market_id: int) -> Dict[str, Any]:
        agent = self.conn.execute(
            "SELECT belief, rho, personality FROM agents WHERE id = ?",
            (agent_id,),
        ).fetchone()
        row = self.conn.execute(
            "SELECT * FROM positions WHERE agent_id = ? AND market_id = ?",
            (agent_id, market_id),
        ).fetchone()
        if row is None:
            return {
                "agent_id": agent_id, "market_id": market_id,
                "yes_shares": 0.0,
                "belief": agent["belief"] if agent is not None else None,
                "rho": agent["rho"] if agent is not None else None,
                "personality": agent["personality"] if agent is not None else None,
            }
        return {
            "agent_id": row["agent_id"],
            "market_id": row["market_id"],
            "yes_shares": row["yes_shares"],
            "belief": agent["belief"] if agent is not None else None,
            "rho": agent["rho"] if agent is not None else None,
            "personality": agent["personality"] if agent is not None else None,
        }

    # ── LMSR Trading ──────────────────────────────────────────────────

    def submit_trade(
        self,
        agent_id: int,
        market_id: int,
        side: str,
        shares: float,
    ) -> Dict[str, Any]:
        if side not in ("buy_yes", "sell_yes"):
            raise ValueError(f"side must be 'buy_yes' or 'sell_yes', got {side!r}")
        if shares <= 0:
            raise ValueError("shares must be positive")

        with self._transaction():
            mkt = self.conn.execute(
                "SELECT * FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            self._check_tradeable(mkt)
            if mkt["mechanism"] != "lmsr":
                raise ValueError(
                    f"submit_trade is for LMSR markets; market {market_id} uses "
                    f"{mkt['mechanism']!r}. Use submit_limit_order / submit_market_order."
                )

            agent = self.conn.execute(
                "SELECT * FROM agents WHERE id = ? AND deleted_at IS NULL", (agent_id,)
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")

            b = mkt["b"]
            old_yes, old_no = mkt["inv_yes"], mkt["inv_no"]
            price_before = self._lmsr_price(old_yes, old_no, b)
            old_cost = self._lmsr_cost(old_yes, old_no, b)

            if side == "buy_yes":
                new_yes, new_no = old_yes + shares, old_no
                share_delta = shares
            else:
                new_yes, new_no = old_yes - shares, old_no
                share_delta = -shares

            new_cost = self._lmsr_cost(new_yes, new_no, b)
            cost = new_cost - old_cost

            if agent["cash"] < cost:
                raise ValueError(
                    f"Insufficient funds: agent has {agent['cash']:.4f}, "
                    f"trade costs {cost:.4f}"
                )

            price_after = self._lmsr_price(new_yes, new_no, b)

            self.conn.execute(
                "UPDATE markets SET inv_yes = ?, inv_no = ? WHERE id = ?",
                (new_yes, new_no, market_id),
            )
            self.conn.execute(
                "UPDATE agents SET cash = cash - ? WHERE id = ?",
                (cost, agent_id),
            )
            self.conn.execute(
                "INSERT INTO positions (agent_id, market_id, yes_shares) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(agent_id, market_id) "
                "DO UPDATE SET yes_shares = yes_shares + ?",
                (agent_id, market_id, share_delta, share_delta),
            )
            now = datetime.now(timezone.utc).isoformat()
            cur = self.conn.execute(
                "INSERT INTO trades "
                "(market_id, agent_id, side, shares, cost, price_before, price_after, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (market_id, agent_id, side, shares, cost, price_before, price_after, now),
            )

        return {
            "trade_id": cur.lastrowid,
            "side": side,
            "shares": shares,
            "cost": cost,
            "price_before": price_before,
            "price_after": price_after,
        }

    # ── CDA Trading ────────────────────────────────────────────────────

    def submit_limit_order(
        self, agent_id: int, market_id: int, side: str,
        quantity: float, price: float,
    ) -> Dict[str, Any]:
        return self._submit_cda_order(
            agent_id=agent_id, market_id=market_id, side=side,
            quantity=quantity, limit_price=price, is_market=False,
        )

    def submit_market_order(
        self, agent_id: int, market_id: int, side: str, quantity: float,
    ) -> Dict[str, Any]:
        return self._submit_cda_order(
            agent_id=agent_id, market_id=market_id, side=side,
            quantity=quantity, limit_price=None, is_market=True,
        )

    def cancel_agent_orders(self, agent_id: int, market_id: int) -> int:
        with self._transaction():
            cur = self.conn.execute(
                "UPDATE orders SET status = 'cancelled' "
                "WHERE agent_id = ? AND market_id = ? AND status = 'open'",
                (agent_id, market_id),
            )
        return cur.rowcount

    def _submit_cda_order(
        self, *, agent_id: int, market_id: int, side: str,
        quantity: float, limit_price: Optional[float], is_market: bool,
    ) -> Dict[str, Any]:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        with self._transaction():
            mkt = self.conn.execute(
                "SELECT * FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            self._check_tradeable(mkt)
            if mkt["mechanism"] != "cda":
                raise ValueError(
                    f"CDA order methods are for CDA markets; market {market_id} uses "
                    f"{mkt['mechanism']!r}. Use submit_trade."
                )

            agent = self.conn.execute(
                "SELECT * FROM agents WHERE id = ? AND deleted_at IS NULL", (agent_id,)
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")

            tick_size = mkt["tick_size"]
            min_p, max_p = mkt["min_price"], mkt["max_price"]
            if limit_price is not None:
                limit_price = self._normalize_price(limit_price, tick_size, min_p, max_p)

            price_before = self._cda_reference_price(market_id)
            remaining = float(quantity)
            matched_trades: List[Dict[str, Any]] = []
            eps = 1e-12
            now = datetime.now(timezone.utc).isoformat()

            while remaining > eps:
                if side == "buy":
                    best = self._cda_best_ask(market_id)
                    if best is None:
                        break
                    if not is_market and limit_price is not None and limit_price < best:
                        break
                    resting = self.conn.execute(
                        "SELECT * FROM orders "
                        "WHERE market_id = ? AND side = 'sell' AND status = 'open' "
                        "  AND price = ? ORDER BY id ASC LIMIT 1",
                        (market_id, best),
                    ).fetchone()
                    if resting is None:
                        break
                    executed = min(remaining, resting["remaining"])
                    trade_price = resting["price"]
                    notional = trade_price * executed
                    buyer_cash = self.conn.execute(
                        "SELECT cash FROM agents WHERE id = ?", (agent_id,)
                    ).fetchone()["cash"]
                    if buyer_cash < notional:
                        break
                    self.conn.execute("UPDATE agents SET cash = cash - ? WHERE id = ?", (notional, agent_id))
                    self.conn.execute("UPDATE agents SET cash = cash + ? WHERE id = ?", (notional, resting["agent_id"]))
                    self.conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) DO UPDATE SET yes_shares = yes_shares + ?",
                        (agent_id, market_id, executed, executed),
                    )
                    self.conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) DO UPDATE SET yes_shares = yes_shares + ?",
                        (resting["agent_id"], market_id, -executed, -executed),
                    )
                    new_rem = resting["remaining"] - executed
                    if new_rem <= eps:
                        self.conn.execute("UPDATE orders SET remaining = 0, status = 'filled' WHERE id = ?", (resting["id"],))
                    else:
                        self.conn.execute("UPDATE orders SET remaining = ? WHERE id = ?", (new_rem, resting["id"]))
                    self.conn.execute("UPDATE markets SET last_trade_price = ? WHERE id = ?", (trade_price, market_id))
                    cur = self.conn.execute(
                        "INSERT INTO trades (market_id, agent_id, side, shares, cost, price_before, price_after, created_at) "
                        "VALUES (?, ?, 'buy', ?, ?, ?, ?, ?)",
                        (market_id, agent_id, executed, notional, trade_price, trade_price, now),
                    )
                    matched_trades.append({
                        "trade_id": cur.lastrowid, "buyer_id": agent_id,
                        "seller_id": resting["agent_id"], "price": trade_price,
                        "quantity": executed, "aggressor_side": "buy",
                    })
                    remaining -= executed
                else:
                    best = self._cda_best_bid(market_id)
                    if best is None:
                        break
                    if not is_market and limit_price is not None and limit_price > best:
                        break
                    resting = self.conn.execute(
                        "SELECT * FROM orders "
                        "WHERE market_id = ? AND side = 'buy' AND status = 'open' "
                        "  AND price = ? ORDER BY id ASC LIMIT 1",
                        (market_id, best),
                    ).fetchone()
                    if resting is None:
                        break
                    executed = min(remaining, resting["remaining"])
                    trade_price = resting["price"]
                    notional = trade_price * executed
                    resting_buyer_cash = self.conn.execute(
                        "SELECT cash FROM agents WHERE id = ?", (resting["agent_id"],)
                    ).fetchone()["cash"]
                    if resting_buyer_cash < notional:
                        self.conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (resting["id"],))
                        continue
                    self.conn.execute("UPDATE agents SET cash = cash + ? WHERE id = ?", (notional, agent_id))
                    self.conn.execute("UPDATE agents SET cash = cash - ? WHERE id = ?", (notional, resting["agent_id"]))
                    self.conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) DO UPDATE SET yes_shares = yes_shares + ?",
                        (agent_id, market_id, -executed, -executed),
                    )
                    self.conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) DO UPDATE SET yes_shares = yes_shares + ?",
                        (resting["agent_id"], market_id, executed, executed),
                    )
                    new_rem = resting["remaining"] - executed
                    if new_rem <= eps:
                        self.conn.execute("UPDATE orders SET remaining = 0, status = 'filled' WHERE id = ?", (resting["id"],))
                    else:
                        self.conn.execute("UPDATE orders SET remaining = ? WHERE id = ?", (new_rem, resting["id"]))
                    self.conn.execute("UPDATE markets SET last_trade_price = ? WHERE id = ?", (trade_price, market_id))
                    cur = self.conn.execute(
                        "INSERT INTO trades (market_id, agent_id, side, shares, cost, price_before, price_after, created_at) "
                        "VALUES (?, ?, 'sell', ?, ?, ?, ?, ?)",
                        (market_id, agent_id, executed, notional, trade_price, trade_price, now),
                    )
                    matched_trades.append({
                        "trade_id": cur.lastrowid, "buyer_id": resting["agent_id"],
                        "seller_id": agent_id, "price": trade_price,
                        "quantity": executed, "aggressor_side": "sell",
                    })
                    remaining -= executed

            resting_order_id = None
            if not is_market and remaining > eps and limit_price is not None:
                cur = self.conn.execute(
                    "INSERT INTO orders (market_id, agent_id, side, price, quantity, remaining, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (market_id, agent_id, side, limit_price, quantity, remaining, now),
                )
                resting_order_id = cur.lastrowid
            price_after = self._cda_reference_price(market_id)

        return {
            "trades": matched_trades, "filled_quantity": quantity - remaining,
            "remaining_quantity": remaining, "resting_order_id": resting_order_id,
            "price_before": price_before, "price_after": price_after,
        }

    # ── Trade history ──────────────────────────────────────────────────

    def get_trades(
        self,
        market_id: Optional[int] = None,
        agent_id: Optional[int] = None,
        since_trade_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if market_id is not None:
            clauses.append("market_id = ?")
            params.append(market_id)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if since_trade_id is not None:
            clauses.append("id > ?")
            params.append(since_trade_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self.conn.execute(
            f"SELECT * FROM trades{where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]

    def create_news_event(
        self,
        *,
        market_id: int,
        headline: str,
        mode: str,
        requested_new_belief: Optional[float],
        requested_delta: Optional[float],
        affected_fraction: float,
        min_signal_sensitivity: float,
        n_candidates: int,
        n_affected: int,
        mean_belief_before: float,
        mean_belief_after: float,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._transaction():
            mkt = self.conn.execute("SELECT id FROM markets WHERE id = ?", (market_id,)).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            cur = self.conn.execute(
                """
                INSERT INTO news_events (
                    market_id, headline, mode, requested_new_belief, requested_delta,
                    affected_fraction, min_signal_sensitivity, n_candidates, n_affected,
                    mean_belief_before, mean_belief_after, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id, headline, mode, requested_new_belief, requested_delta,
                    affected_fraction, min_signal_sensitivity, n_candidates, n_affected,
                    mean_belief_before, mean_belief_after, now,
                ),
            )
            row = self.conn.execute(
                "SELECT * FROM news_events WHERE id = ?",
                (cur.lastrowid,),
            ).fetchone()
        return self._news_row_to_dict(row)

    def list_news_events(
        self,
        market_id: int,
        *,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        mkt = self.conn.execute("SELECT id FROM markets WHERE id = ?", (market_id,)).fetchone()
        if mkt is None:
            raise ValueError(f"Market {market_id} not found")
        total_row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM news_events WHERE market_id = ?",
            (market_id,),
        ).fetchone()
        total = int(total_row["c"] if total_row is not None else 0)
        rows = self.conn.execute(
            """
            SELECT * FROM news_events
            WHERE market_id = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (market_id, limit, offset),
        ).fetchall()
        return {
            "events": [self._news_row_to_dict(r) for r in rows],
            "total": total,
        }

    # ── Row converters ─────────────────────────────────────────────────

    @staticmethod
    def _market_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "slug": row["slug"],
            "title": row["title"],
            "description": row["description"],
            "mechanism": row["mechanism"],
            "ground_truth": row["ground_truth"],
            "b": row["b"],
            "inv_yes": row["inv_yes"],
            "inv_no": row["inv_no"],
            "tick_size": row["tick_size"],
            "min_price": row["min_price"],
            "max_price": row["max_price"],
            "last_trade_price": row["last_trade_price"],
            "initial_price": row["initial_price"],
            "status": row["status"],
            "resolution": row["resolution"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    @staticmethod
    def _agent_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "cash": row["cash"],
            "belief": row["belief"],
            "rho": row["rho"],
            "personality": row["personality"],
            "created_at": row["created_at"],
            "deleted_at": row["deleted_at"],
        }

    @staticmethod
    def _news_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "market_id": row["market_id"],
            "headline": row["headline"],
            "mode": row["mode"],
            "requested_new_belief": row["requested_new_belief"],
            "requested_delta": row["requested_delta"],
            "affected_fraction": row["affected_fraction"],
            "min_signal_sensitivity": row["min_signal_sensitivity"],
            "n_candidates": row["n_candidates"],
            "n_affected": row["n_affected"],
            "mean_belief_before": row["mean_belief_before"],
            "mean_belief_after": row["mean_belief_after"],
            "at_timestamp": row["created_at"],
        }
