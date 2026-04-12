"""
Thread-safe market service with transactional trade execution for LMSR and CDA.

Wraps a file-backed SQLite database with per-thread connections and
``BEGIN IMMEDIATE`` transactions so concurrent callers serialize properly:
the second thread blocks until the first commits, preventing stale-read
races on inventory / cash.

Key differences from MarketStore (single-thread, single-connection):
  * ``threading.local()`` gives each thread its own ``sqlite3.Connection``.
  * All writes use explicit ``BEGIN IMMEDIATE`` → ``COMMIT`` (manual
    ``isolation_level = None``) so the RESERVED lock is held *before*
    reading state, not after.
  * ``PRAGMA busy_timeout`` lets a blocked thread wait rather than fail.

Usage:
    svc = MarketService("markets.db")

    mkt   = svc.create_market("btc-100k", "BTC > $100k?", mechanism="lmsr", b=100.0)
    alice = svc.create_agent("alice", cash=1000.0)

    # safe to call from any thread
    trade = svc.execute_trade(alice["id"], mkt["id"], "buy_yes", shares=5.0)
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from market_store import MarketStore, _SCHEMA

_BUSY_TIMEOUT_MS = 5000


class MarketService:
    """Thread-safe market service backed by a shared SQLite database."""

    def __init__(self, db_path: str):
        if db_path == ":memory:":
            raise ValueError(
                "In-memory databases cannot be shared across threads. "
                "Use a file path or 'file::memory:?cache=shared' with uri=True."
            )
        self._db_path = db_path
        self._uri = db_path.startswith("file:")
        self._local = threading.local()

        conn = self._get_conn()
        conn.executescript(_SCHEMA)

    def _get_conn(self) -> sqlite3.Connection:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self._db_path,
                uri=self._uri,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            conn.isolation_level = None  # manual transaction control
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ── Markets ────────────────────────────────────────────────────────

    def create_market(
        self,
        slug: str,
        title: str,
        *,
        mechanism: str = "lmsr",
        b: Optional[float] = None,
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

        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "INSERT INTO markets "
                "(slug, title, description, mechanism, b, tick_size, min_price, max_price, "
                " initial_price, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (slug, title, description, mechanism, b, tick_size, min_price, max_price,
                 initial_price, now),
            )
            row = conn.execute(
                "SELECT * FROM markets WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            conn.execute("COMMIT")
            return MarketStore._market_row_to_dict(row)
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def get_market(self, market_id: int) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM markets WHERE id = ?", (market_id,)).fetchone()
        if row is None:
            raise ValueError(f"Market {market_id} not found")
        return MarketStore._market_row_to_dict(row)

    def list_markets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM markets WHERE status = ? ORDER BY id", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM markets ORDER BY id").fetchall()
        return [MarketStore._market_row_to_dict(r) for r in rows]

    def get_price(self, market_id: int) -> float:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT mechanism, inv_yes, inv_no, b FROM markets WHERE id = ?",
            (market_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Market {market_id} not found")
        if row["mechanism"] == "lmsr":
            return MarketStore._lmsr_price(row["inv_yes"], row["inv_no"], row["b"])
        return self._cda_reference_price(conn, market_id)

    def get_order_book(self, market_id: int) -> Dict[str, Any]:
        conn = self._get_conn()
        bids = conn.execute(
            "SELECT price, SUM(remaining) as total_qty FROM orders "
            "WHERE market_id = ? AND side = 'buy' AND status = 'open' "
            "GROUP BY price ORDER BY price DESC",
            (market_id,),
        ).fetchall()
        asks = conn.execute(
            "SELECT price, SUM(remaining) as total_qty FROM orders "
            "WHERE market_id = ? AND side = 'sell' AND status = 'open' "
            "GROUP BY price ORDER BY price ASC",
            (market_id,),
        ).fetchall()
        return {
            "bids": [{"price": r["price"], "quantity": r["total_qty"]} for r in bids],
            "asks": [{"price": r["price"], "quantity": r["total_qty"]} for r in asks],
            "best_bid": self._cda_best_bid(conn, market_id),
            "best_ask": self._cda_best_ask(conn, market_id),
        }

    def resolve_market(self, market_id: int, outcome: str) -> Dict[str, Any]:
        if outcome not in ("yes", "no"):
            raise ValueError(f"outcome must be 'yes' or 'no', got {outcome!r}")
        payoff = 1.0 if outcome == "yes" else 0.0
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            mkt = conn.execute("SELECT * FROM markets WHERE id = ?", (market_id,)).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            if mkt["status"] == "resolved":
                raise ValueError(f"Market {market_id} is already resolved")
            conn.execute(
                "UPDATE orders SET status = 'cancelled' "
                "WHERE market_id = ? AND status = 'open'",
                (market_id,),
            )
            conn.execute(
                "UPDATE markets SET status = 'resolved', resolution = ?, resolved_at = ? "
                "WHERE id = ?",
                (outcome, now, market_id),
            )
            positions = conn.execute(
                "SELECT agent_id, yes_shares FROM positions WHERE market_id = ?",
                (market_id,),
            ).fetchall()
            for pos in positions:
                payout = pos["yes_shares"] * payoff
                conn.execute(
                    "UPDATE agents SET cash = cash + ? WHERE id = ?",
                    (payout, pos["agent_id"]),
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return {
            "market_id": market_id,
            "outcome": outcome,
            "positions_settled": len(positions),
        }

    # ── Agents ─────────────────────────────────────────────────────────

    def create_agent(self, name: str, cash: float) -> Dict[str, Any]:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "INSERT INTO agents (name, cash, created_at) VALUES (?, ?, ?)",
                (name, cash, now),
            )
            row = conn.execute(
                "SELECT * FROM agents WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
            conn.execute("COMMIT")
            return MarketStore._agent_row_to_dict(row)
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def get_agent(self, agent_id: int) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if row is None:
            raise ValueError(f"Agent {agent_id} not found")
        return MarketStore._agent_row_to_dict(row)

    def get_position(self, agent_id: int, market_id: int) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM positions WHERE agent_id = ? AND market_id = ?",
            (agent_id, market_id),
        ).fetchone()
        if row is None:
            return {"agent_id": agent_id, "market_id": market_id, "yes_shares": 0.0}
        return {
            "agent_id": row["agent_id"],
            "market_id": row["market_id"],
            "yes_shares": row["yes_shares"],
        }

    # ── LMSR Trading ──────────────────────────────────────────────────

    def execute_trade(
        self,
        agent_id: int,
        market_id: int,
        side: str,
        shares: float,
    ) -> Dict[str, Any]:
        """
        Thread-safe LMSR trade execution.

        Acquires a write lock (BEGIN IMMEDIATE) *before* reading inventory
        so concurrent callers serialize: the second thread blocks until the
        first commits, guaranteeing both see up-to-date state.
        """
        if side not in ("buy_yes", "sell_yes"):
            raise ValueError(f"side must be 'buy_yes' or 'sell_yes', got {side!r}")
        if shares <= 0:
            raise ValueError("shares must be positive")

        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            mkt = conn.execute(
                "SELECT * FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            if mkt["status"] != "open":
                raise ValueError(f"Market {market_id} is {mkt['status']}, not open")
            if mkt["mechanism"] != "lmsr":
                raise ValueError(
                    f"execute_trade is for LMSR markets; market {market_id} uses "
                    f"{mkt['mechanism']!r}. Use execute_limit_order / execute_market_order."
                )

            agent = conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")

            b = mkt["b"]
            old_yes, old_no = mkt["inv_yes"], mkt["inv_no"]
            price_before = MarketStore._lmsr_price(old_yes, old_no, b)
            old_cost = MarketStore._lmsr_cost(old_yes, old_no, b)

            if side == "buy_yes":
                new_yes, new_no = old_yes + shares, old_no
                share_delta = shares
            else:
                new_yes, new_no = old_yes - shares, old_no
                share_delta = -shares

            new_cost = MarketStore._lmsr_cost(new_yes, new_no, b)
            cost = new_cost - old_cost

            if agent["cash"] < cost:
                raise ValueError(
                    f"Insufficient funds: agent has {agent['cash']:.4f}, "
                    f"trade costs {cost:.4f}"
                )

            price_after = MarketStore._lmsr_price(new_yes, new_no, b)

            conn.execute(
                "UPDATE markets SET inv_yes = ?, inv_no = ? WHERE id = ?",
                (new_yes, new_no, market_id),
            )
            conn.execute(
                "UPDATE agents SET cash = cash - ? WHERE id = ?",
                (cost, agent_id),
            )
            conn.execute(
                "INSERT INTO positions (agent_id, market_id, yes_shares) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(agent_id, market_id) "
                "DO UPDATE SET yes_shares = yes_shares + ?",
                (agent_id, market_id, share_delta, share_delta),
            )
            now = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                "INSERT INTO trades "
                "(market_id, agent_id, side, shares, cost, price_before, price_after, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (market_id, agent_id, side, shares, cost, price_before, price_after, now),
            )
            trade_id = cur.lastrowid
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return {
            "trade_id": trade_id,
            "side": side,
            "shares": shares,
            "cost": cost,
            "price_before": price_before,
            "price_after": price_after,
        }

    # ── CDA Trading ────────────────────────────────────────────────────

    def execute_limit_order(
        self,
        agent_id: int,
        market_id: int,
        side: str,
        quantity: float,
        price: float,
    ) -> Dict[str, Any]:
        """Thread-safe CDA limit order. Crosses spread if price-improving."""
        return self._execute_cda_order(
            agent_id=agent_id,
            market_id=market_id,
            side=side,
            quantity=quantity,
            limit_price=price,
            is_market=False,
        )

    def execute_market_order(
        self,
        agent_id: int,
        market_id: int,
        side: str,
        quantity: float,
    ) -> Dict[str, Any]:
        """Thread-safe CDA market order. Walks the book, never rests."""
        return self._execute_cda_order(
            agent_id=agent_id,
            market_id=market_id,
            side=side,
            quantity=quantity,
            limit_price=None,
            is_market=True,
        )

    def cancel_agent_orders(self, agent_id: int, market_id: int) -> int:
        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            cur = conn.execute(
                "UPDATE orders SET status = 'cancelled' "
                "WHERE agent_id = ? AND market_id = ? AND status = 'open'",
                (agent_id, market_id),
            )
            count = cur.rowcount
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return count

    def _execute_cda_order(
        self,
        *,
        agent_id: int,
        market_id: int,
        side: str,
        quantity: float,
        limit_price: Optional[float],
        is_market: bool,
    ) -> Dict[str, Any]:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        conn = self._get_conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            mkt = conn.execute(
                "SELECT * FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            if mkt["status"] != "open":
                raise ValueError(f"Market {market_id} is {mkt['status']}, not open")
            if mkt["mechanism"] != "cda":
                raise ValueError(
                    f"CDA order methods are for CDA markets; market {market_id} uses "
                    f"{mkt['mechanism']!r}. Use execute_trade."
                )

            agent = conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")

            tick_size = mkt["tick_size"]
            min_p, max_p = mkt["min_price"], mkt["max_price"]

            if limit_price is not None:
                clipped = min(max_p, max(min_p, float(limit_price)))
                limit_price = round(clipped / tick_size) * tick_size

            price_before = self._cda_reference_price(conn, market_id)
            remaining = float(quantity)
            matched_trades: List[Dict[str, Any]] = []
            eps = 1e-12
            now = datetime.now(timezone.utc).isoformat()

            while remaining > eps:
                if side == "buy":
                    best = self._cda_best_ask(conn, market_id)
                    if best is None:
                        break
                    if not is_market and limit_price is not None and limit_price < best:
                        break
                    resting = conn.execute(
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

                    buyer_cash = conn.execute(
                        "SELECT cash FROM agents WHERE id = ?", (agent_id,)
                    ).fetchone()["cash"]
                    if buyer_cash < notional:
                        break

                    conn.execute(
                        "UPDATE agents SET cash = cash - ? WHERE id = ?",
                        (notional, agent_id),
                    )
                    conn.execute(
                        "UPDATE agents SET cash = cash + ? WHERE id = ?",
                        (notional, resting["agent_id"]),
                    )
                    conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) "
                        "DO UPDATE SET yes_shares = yes_shares + ?",
                        (agent_id, market_id, executed, executed),
                    )
                    conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) "
                        "DO UPDATE SET yes_shares = yes_shares + ?",
                        (resting["agent_id"], market_id, -executed, -executed),
                    )

                    new_rem = resting["remaining"] - executed
                    if new_rem <= eps:
                        conn.execute(
                            "UPDATE orders SET remaining = 0, status = 'filled' WHERE id = ?",
                            (resting["id"],),
                        )
                    else:
                        conn.execute(
                            "UPDATE orders SET remaining = ? WHERE id = ?",
                            (new_rem, resting["id"]),
                        )

                    conn.execute(
                        "UPDATE markets SET last_trade_price = ? WHERE id = ?",
                        (trade_price, market_id),
                    )
                    cur = conn.execute(
                        "INSERT INTO trades "
                        "(market_id, agent_id, side, shares, cost, "
                        " price_before, price_after, created_at) "
                        "VALUES (?, ?, 'buy', ?, ?, ?, ?, ?)",
                        (market_id, agent_id, executed, notional,
                         trade_price, trade_price, now),
                    )
                    matched_trades.append({
                        "trade_id": cur.lastrowid,
                        "buyer_id": agent_id,
                        "seller_id": resting["agent_id"],
                        "price": trade_price,
                        "quantity": executed,
                        "aggressor_side": "buy",
                    })
                    remaining -= executed

                else:  # sell
                    best = self._cda_best_bid(conn, market_id)
                    if best is None:
                        break
                    if not is_market and limit_price is not None and limit_price > best:
                        break
                    resting = conn.execute(
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

                    resting_buyer_cash = conn.execute(
                        "SELECT cash FROM agents WHERE id = ?", (resting["agent_id"],)
                    ).fetchone()["cash"]
                    if resting_buyer_cash < notional:
                        conn.execute(
                            "UPDATE orders SET status = 'cancelled' WHERE id = ?",
                            (resting["id"],),
                        )
                        continue

                    conn.execute(
                        "UPDATE agents SET cash = cash + ? WHERE id = ?",
                        (notional, agent_id),
                    )
                    conn.execute(
                        "UPDATE agents SET cash = cash - ? WHERE id = ?",
                        (notional, resting["agent_id"]),
                    )
                    conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) "
                        "DO UPDATE SET yes_shares = yes_shares + ?",
                        (agent_id, market_id, -executed, -executed),
                    )
                    conn.execute(
                        "INSERT INTO positions (agent_id, market_id, yes_shares) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT(agent_id, market_id) "
                        "DO UPDATE SET yes_shares = yes_shares + ?",
                        (resting["agent_id"], market_id, executed, executed),
                    )

                    new_rem = resting["remaining"] - executed
                    if new_rem <= eps:
                        conn.execute(
                            "UPDATE orders SET remaining = 0, status = 'filled' WHERE id = ?",
                            (resting["id"],),
                        )
                    else:
                        conn.execute(
                            "UPDATE orders SET remaining = ? WHERE id = ?",
                            (new_rem, resting["id"]),
                        )

                    conn.execute(
                        "UPDATE markets SET last_trade_price = ? WHERE id = ?",
                        (trade_price, market_id),
                    )
                    cur = conn.execute(
                        "INSERT INTO trades "
                        "(market_id, agent_id, side, shares, cost, "
                        " price_before, price_after, created_at) "
                        "VALUES (?, ?, 'sell', ?, ?, ?, ?, ?)",
                        (market_id, agent_id, executed, notional,
                         trade_price, trade_price, now),
                    )
                    matched_trades.append({
                        "trade_id": cur.lastrowid,
                        "buyer_id": resting["agent_id"],
                        "seller_id": agent_id,
                        "price": trade_price,
                        "quantity": executed,
                        "aggressor_side": "sell",
                    })
                    remaining -= executed

            resting_order_id = None
            if not is_market and remaining > eps and limit_price is not None:
                cur = conn.execute(
                    "INSERT INTO orders "
                    "(market_id, agent_id, side, price, quantity, remaining, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (market_id, agent_id, side, limit_price, quantity, remaining, now),
                )
                resting_order_id = cur.lastrowid

            price_after = self._cda_reference_price(conn, market_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return {
            "trades": matched_trades,
            "filled_quantity": quantity - remaining,
            "remaining_quantity": remaining,
            "resting_order_id": resting_order_id,
            "price_before": price_before,
            "price_after": price_after,
        }

    # ── Trade history ──────────────────────────────────────────────────

    def get_trades(
        self,
        market_id: Optional[int] = None,
        agent_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        clauses: List[str] = []
        params: List[Any] = []
        if market_id is not None:
            clauses.append("market_id = ?")
            params.append(market_id)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM trades{where} ORDER BY id DESC LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]

    # ── CDA helpers (take explicit conn for use inside transactions) ──

    @staticmethod
    def _cda_best_bid(conn: sqlite3.Connection, market_id: int) -> Optional[float]:
        row = conn.execute(
            "SELECT MAX(price) AS p FROM orders "
            "WHERE market_id = ? AND side = 'buy' AND status = 'open'",
            (market_id,),
        ).fetchone()
        return row["p"] if row and row["p"] is not None else None

    @staticmethod
    def _cda_best_ask(conn: sqlite3.Connection, market_id: int) -> Optional[float]:
        row = conn.execute(
            "SELECT MIN(price) AS p FROM orders "
            "WHERE market_id = ? AND side = 'sell' AND status = 'open'",
            (market_id,),
        ).fetchone()
        return row["p"] if row and row["p"] is not None else None

    @staticmethod
    def _cda_reference_price(conn: sqlite3.Connection, market_id: int) -> float:
        bid = MarketService._cda_best_bid(conn, market_id)
        ask = MarketService._cda_best_ask(conn, market_id)
        if bid is not None and ask is not None:
            return 0.5 * (bid + ask)
        mkt = conn.execute(
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
