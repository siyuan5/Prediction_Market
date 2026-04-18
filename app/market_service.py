"""
Thread-safe market service with transactional trade execution for LMSR and CDA.

Depends on MarketStore for persistence and row conversion, LMSRMarketMaker
(team_a) for LMSR cost-function math, and ContinuousDoubleAuction (team_b)
for CDA order matching.  Does not re-implement either pricing or matching
logic.

Usage:
    svc = MarketService("markets.db")
    mkt   = svc.create_market("btc-100k", "BTC > $100k?", mechanism="lmsr", b=100.0)
    alice = svc.create_agent("alice", cash=1000.0)
    svc.set_market_status(mkt["id"], "running")
    trade = svc.execute_lmsr_trade(mkt["id"], alice["id"], quantity=5.0)
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from team_a_market_logic import LMSRMarketMaker
from team_b_market_logic import ContinuousDoubleAuction, Trade

from market_store import MarketStore, _SCHEMA, _TRADEABLE_STATUSES

_BUSY_TIMEOUT_MS = 5000


class MarketService:
    """Thread-safe market service backed by a shared SQLite database.

    Creates per-thread connections and per-thread ``MarketStore`` instances
    (with ``_external_transactions=True``) so that all write operations are
    serialized via ``BEGIN IMMEDIATE``.
    """

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

    # ── Connection / store management ─────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, uri=self._uri, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.isolation_level = None
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _get_store(self) -> MarketStore:
        store: Optional[MarketStore] = getattr(self._local, "store", None)
        if store is None:
            store = MarketStore(
                _conn=self._get_conn(), _external_transactions=True,
            )
            self._local.store = store
        return store

    @contextmanager
    def _begin_immediate(self):
        """Context manager: wraps body in BEGIN IMMEDIATE / COMMIT / ROLLBACK."""
        conn = self._get_store().conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
            self._local.store = None

    # ── Read operations (delegate to MarketStore) ─────────────────────

    def get_market(self, market_id: int) -> Dict[str, Any]:
        return self._get_store().get_market(market_id)

    def list_markets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._get_store().list_markets(status)

    def list_markets_with_summary(
        self,
        status: Optional[str] = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Market-discovery helper for API/UI.

        Returns paged market rows with current price, total trade count, and
        active agents (distinct traders) derived from persisted trade history.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        all_markets = self._get_store().list_markets(status)
        total = len(all_markets)
        page = all_markets[offset : offset + limit]
        out: List[Dict[str, Any]] = []
        store = self._get_store()
        for m in page:
            mid = int(m["id"])
            row = store.conn.execute(
                """
                SELECT
                    COUNT(*) AS trade_count,
                    COUNT(DISTINCT agent_id) AS active_agents
                FROM trades
                WHERE market_id = ?
                """,
                (mid,),
            ).fetchone()
            out.append(
                {
                    **m,
                    "price": self.get_price(mid),
                    "trade_count": int(row["trade_count"] if row else 0),
                    "active_agents": int(row["active_agents"] if row else 0),
                }
            )
        return {"markets": out, "total": total}

    def get_agent(self, agent_id: int, market_id: Optional[int] = None) -> Dict[str, Any]:
        agent = self._get_store().get_agent(agent_id)
        if market_id is not None:
            pos = self._get_store().get_position(agent_id, market_id)
            agent["yes_shares"] = pos["yes_shares"]
        return agent

    def list_agents(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._get_store().list_agents(limit=limit, offset=offset)

    def get_position(self, agent_id: int, market_id: int) -> Dict[str, Any]:
        return self._get_store().get_position(agent_id, market_id)

    def get_order_book(self, market_id: int) -> Dict[str, Any]:
        return self._get_store().get_order_book(market_id)

    def get_trades(
        self, market_id: Optional[int] = None, agent_id: Optional[int] = None,
        since_trade_id: Optional[int] = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self._get_store().get_trades(market_id, agent_id, since_trade_id, limit)

    def count_trades(self, market_id: int) -> int:
        row = self._get_store().conn.execute(
            "SELECT COUNT(*) AS c FROM trades WHERE market_id = ?", (market_id,)
        ).fetchone()
        return int(row["c"] if hasattr(row, "keys") else row[0])

    def list_agents_for_market(self, market_id: int) -> List[Dict[str, Any]]:
        """
        Return one row per agent with a position in this market: id, name, cash,
        yes_shares, belief, rho, personality (raw string from DB).
        """
        store = self._get_store()
        rows = store.conn.execute(
            """
            SELECT a.id AS agent_id, a.name, a.cash, a.belief, a.rho, a.personality,
                   p.yes_shares
            FROM positions p
            JOIN agents a ON a.id = p.agent_id
            WHERE p.market_id = ?
            ORDER BY a.id
            """,
            (market_id,),
        ).fetchall()
        return [{k: r[k] for k in r.keys()} for r in rows]

    # ── Write operations (delegate with BEGIN IMMEDIATE) ──────────────

    def create_market(
        self, slug: str, title: str, *, mechanism: str = "lmsr",
        b: Optional[float] = None, ground_truth: Optional[float] = None,
        description: str = "", tick_size: float = 0.0001,
        min_price: float = 0.001, max_price: float = 0.999,
        initial_price: float = 0.5,
    ) -> Dict[str, Any]:
        with self._begin_immediate():
            return self._get_store().create_market(
                slug, title, mechanism=mechanism, b=b, ground_truth=ground_truth,
                description=description, tick_size=tick_size,
                min_price=min_price, max_price=max_price,
                initial_price=initial_price,
            )

    def create_agent(
        self, name: str, cash: float, *, market_id: Optional[int] = None,
        belief: Optional[float] = None, rho: Optional[float] = None,
        personality: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._begin_immediate():
            agent = self._get_store().create_agent(
                name, cash, belief=belief,
                rho=rho, personality=personality,
            )
            if market_id is not None:
                self._get_store().ensure_position(agent["id"], market_id)
            return agent

    def set_market_status(self, market_id: int, status: str) -> Dict[str, Any]:
        with self._begin_immediate():
            return self._get_store().set_market_status(market_id, status)

    def set_agent_belief(self, market_id: int, agent_id: int, new_belief: float) -> float:
        with self._begin_immediate():
            return self._get_store().set_agent_belief(agent_id, new_belief)

    def update_agent_portfolio(
        self, market_id: int, agent_id: int, cash_delta: float, shares_delta: float,
    ) -> Dict[str, Any]:
        with self._begin_immediate():
            return self._get_store().update_agent_portfolio(
                market_id, agent_id, cash_delta, shares_delta,
            )

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
        with self._begin_immediate():
            return self._get_store().update_agent(
                agent_id,
                name=name,
                cash=cash,
                belief=belief,
                rho=rho,
                personality=personality,
            )

    def ensure_position(self, agent_id: int, market_id: int) -> Dict[str, Any]:
        """Lazy-position helper used by trade execution paths."""
        with self._begin_immediate():
            return self._get_store().ensure_position(agent_id, market_id)

    def resolve_market(self, market_id: int, outcome: str) -> Dict[str, Any]:
        with self._begin_immediate():
            return self._get_store().resolve_market(market_id, outcome)

    def cancel_agent_orders(self, agent_id: int, market_id: int) -> int:
        with self._begin_immediate():
            return self._get_store().cancel_agent_orders(agent_id, market_id)

    # ── Pricing (uses LMSRMarketMaker for team_a delegation) ──────────

    def get_price(self, market_id: int) -> float:
        store = self._get_store()
        row = store.conn.execute(
            "SELECT mechanism, inv_yes, inv_no, b FROM markets WHERE id = ?",
            (market_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Market {market_id} not found")
        if row["mechanism"] == "lmsr":
            mm = LMSRMarketMaker(row["b"], [row["inv_yes"], row["inv_no"]])
            return float(mm.get_price())
        return store._cda_reference_price(market_id)

    def get_price_snapshot(self, market_id: int) -> Dict[str, Any]:
        store = self._get_store()
        mkt = store.get_market(market_id)
        result: Dict[str, Any] = {
            "market_id": market_id,
            "mechanism": mkt["mechanism"],
            "status": mkt["status"],
        }
        if mkt["mechanism"] == "lmsr":
            mm = LMSRMarketMaker(mkt["b"], [mkt["inv_yes"], mkt["inv_no"]])
            result["price"] = float(mm.get_price())
            result["inv_yes"] = mkt["inv_yes"]
            result["inv_no"] = mkt["inv_no"]
            result["b"] = mkt["b"]
        else:
            result["price"] = store._cda_reference_price(market_id)
            result["best_bid"] = store._cda_best_bid(market_id)
            result["best_ask"] = store._cda_best_ask(market_id)
            result["last_trade_price"] = mkt["last_trade_price"]
        return result

    # ── LMSR Trading ──────────────────────────────────────────────────

    def execute_lmsr_trade(
        self, market_id: int, agent_id: int, quantity: float,
    ) -> Dict[str, Any]:
        """
        Execute an LMSR trade.  Positive quantity = buy YES, negative = sell YES.
        Clips to the largest affordable quantity when the agent has insufficient cash.
        Uses LMSRMarketMaker from team_a_market_logic for all cost/price math.
        """
        if quantity == 0:
            raise ValueError("quantity must be non-zero")
        store = self._get_store()
        conn = store.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            mkt = conn.execute(
                "SELECT * FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            store._check_tradeable(mkt)
            if mkt["mechanism"] != "lmsr":
                raise ValueError(
                    f"execute_lmsr_trade is for LMSR markets; "
                    f"market {market_id} uses {mkt['mechanism']!r}."
                )
            agent = conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")
            # Explicit lazy-link helper per Task 2; safe under active tx.
            store.ensure_position(agent_id, market_id)

            b = mkt["b"]
            inv_yes, inv_no = float(mkt["inv_yes"]), float(mkt["inv_no"])

            mm = LMSRMarketMaker(b, [inv_yes, inv_no])
            price_before = float(mm.get_price())

            old_cost_val = float(mm.get_cost(mm.inventory))
            full_inv = np.array([inv_yes + quantity, inv_no], dtype=float)
            full_cost = float(mm.get_cost(full_inv)) - old_cost_val

            actual_quantity = quantity
            clipped = False

            if full_cost > agent["cash"] and quantity > 0:
                actual_quantity = self._clip_lmsr_buy(
                    inv_yes, inv_no, b, quantity, agent["cash"],
                )
                clipped = True
                if actual_quantity < 1e-9:
                    conn.execute("ROLLBACK")
                    return {
                        "trade_id": None, "market_id": market_id,
                        "agent_id": agent_id, "quantity": 0.0,
                        "requested_quantity": quantity, "clipped": True,
                        "cost": 0.0, "price_before": price_before,
                        "price_after": price_before,
                    }

            exec_mm = LMSRMarketMaker(b, [inv_yes, inv_no])
            cost = float(exec_mm.calculate_trade_cost(actual_quantity))
            new_inv_yes = float(exec_mm.inventory[0])
            new_inv_no = float(exec_mm.inventory[1])
            price_after = float(exec_mm.get_price())

            side = "buy_yes" if actual_quantity >= 0 else "sell_yes"
            share_delta = actual_quantity

            conn.execute(
                "UPDATE markets SET inv_yes = ?, inv_no = ? WHERE id = ?",
                (new_inv_yes, new_inv_no, market_id),
            )
            conn.execute(
                "UPDATE agents SET cash = cash - ? WHERE id = ?",
                (cost, agent_id),
            )
            conn.execute(
                "INSERT INTO positions (agent_id, market_id, yes_shares) VALUES (?, ?, ?) "
                "ON CONFLICT(agent_id, market_id) DO UPDATE SET yes_shares = yes_shares + ?",
                (agent_id, market_id, share_delta, share_delta),
            )
            now = datetime.now(timezone.utc).isoformat()
            cur = conn.execute(
                "INSERT INTO trades "
                "(market_id, agent_id, side, shares, cost, price_before, price_after, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (market_id, agent_id, side, abs(actual_quantity), cost,
                 price_before, price_after, now),
            )
            trade_id = cur.lastrowid
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return {
            "trade_id": trade_id, "market_id": market_id, "agent_id": agent_id,
            "quantity": actual_quantity, "requested_quantity": quantity,
            "clipped": clipped, "cost": cost,
            "price_before": price_before, "price_after": price_after,
        }

    def execute_trade(
        self, agent_id: int, market_id: int, side: str, shares: float,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper around execute_lmsr_trade."""
        if side not in ("buy_yes", "sell_yes"):
            raise ValueError(f"side must be 'buy_yes' or 'sell_yes', got {side!r}")
        if shares <= 0:
            raise ValueError("shares must be positive")
        quantity = shares if side == "buy_yes" else -shares
        result = self.execute_lmsr_trade(market_id, agent_id, quantity)
        if result["trade_id"] is None:
            raise ValueError(
                f"Insufficient funds: trade fully clipped to zero"
            )
        return {
            "trade_id": result["trade_id"],
            "side": side,
            "shares": abs(result["quantity"]),
            "cost": result["cost"],
            "price_before": result["price_before"],
            "price_after": result["price_after"],
        }

    @staticmethod
    def _clip_lmsr_buy(
        inv_yes: float, inv_no: float, b: float,
        max_quantity: float, max_cost: float,
    ) -> float:
        """Binary-search for the largest quantity whose LMSR cost <= max_cost."""
        mm = LMSRMarketMaker(b, [inv_yes, inv_no])
        old_cost_val = float(mm.get_cost(mm.inventory))
        lo, hi = 0.0, max_quantity
        for _ in range(64):
            mid = (lo + hi) * 0.5
            test_inv = np.array([inv_yes + mid, inv_no], dtype=float)
            cost = float(mm.get_cost(test_inv)) - old_cost_val
            if cost <= max_cost + 1e-12:
                lo = mid
            else:
                hi = mid
        return lo

    # ── CDA Trading ────────────────────────────────────────────────────

    def execute_cda_order(
        self, market_id: int, agent_id: int, side: str,
        quantity: float, limit_price: Optional[float],
        order_type: str,
    ) -> Dict[str, Any]:
        """
        Submit a CDA order.  order_type: 'limit' or 'market'.
        Delegates matching to ContinuousDoubleAuction from team_b_market_logic.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if order_type not in ("limit", "market"):
            raise ValueError(f"order_type must be 'limit' or 'market', got {order_type!r}")

        store = self._get_store()
        conn = store.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            mkt = conn.execute(
                "SELECT * FROM markets WHERE id = ?", (market_id,)
            ).fetchone()
            if mkt is None:
                raise ValueError(f"Market {market_id} not found")
            store._check_tradeable(mkt)
            if mkt["mechanism"] != "cda":
                raise ValueError(
                    f"CDA methods are for CDA markets; "
                    f"market {market_id} uses {mkt['mechanism']!r}."
                )
            agent = conn.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
            if agent is None:
                raise ValueError(f"Agent {agent_id} not found")
            # Ensure the aggressor has a position row before matching/writes.
            store.ensure_position(agent_id, market_id)

            price_before = store._cda_reference_price(market_id)

            cda, resting_lookup = self._hydrate_cda(store, mkt)

            if order_type == "limit":
                cda_result = cda.submit_limit_order(
                    agent_id=agent_id, side=side,
                    quantity=quantity, limit_price=limit_price,
                )
            else:
                cda_result = cda.submit_market_order(
                    agent_id=agent_id, side=side, quantity=quantity,
                )

            persisted_trades: List[Dict[str, Any]] = []
            total_filled = 0.0
            now = datetime.now(timezone.utc).isoformat()

            for trade in cda_result["trades"]:
                fill_qty = trade.quantity
                notional = trade.price * fill_qty

                buyer_cash = conn.execute(
                    "SELECT cash FROM agents WHERE id = ?", (trade.buyer_id,)
                ).fetchone()["cash"]
                if buyer_cash < notional:
                    affordable = buyer_cash / trade.price if trade.price > 0 else 0.0
                    if affordable < 1e-12:
                        break
                    fill_qty = affordable
                    notional = trade.price * fill_qty

                conn.execute(
                    "UPDATE agents SET cash = cash - ? WHERE id = ?",
                    (notional, trade.buyer_id),
                )
                conn.execute(
                    "UPDATE agents SET cash = cash + ? WHERE id = ?",
                    (notional, trade.seller_id),
                )
                store.ensure_position(trade.buyer_id, market_id)
                store.ensure_position(trade.seller_id, market_id)
                conn.execute(
                    "INSERT INTO positions (agent_id, market_id, yes_shares) "
                    "VALUES (?, ?, ?) ON CONFLICT(agent_id, market_id) "
                    "DO UPDATE SET yes_shares = yes_shares + ?",
                    (trade.buyer_id, market_id, fill_qty, fill_qty),
                )
                conn.execute(
                    "INSERT INTO positions (agent_id, market_id, yes_shares) "
                    "VALUES (?, ?, ?) ON CONFLICT(agent_id, market_id) "
                    "DO UPDATE SET yes_shares = yes_shares + ?",
                    (trade.seller_id, market_id, -fill_qty, -fill_qty),
                )
                conn.execute(
                    "UPDATE markets SET last_trade_price = ? WHERE id = ?",
                    (trade.price, market_id),
                )

                cur = conn.execute(
                    "INSERT INTO trades "
                    "(market_id, agent_id, side, shares, cost, "
                    " price_before, price_after, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (market_id, agent_id, trade.aggressor_side, fill_qty,
                     notional, trade.price, trade.price, now),
                )
                persisted_trades.append({
                    "trade_id": cur.lastrowid,
                    "buyer_id": trade.buyer_id,
                    "seller_id": trade.seller_id,
                    "price": trade.price,
                    "quantity": fill_qty,
                    "aggressor_side": trade.aggressor_side,
                })
                total_filled += fill_qty

                if trade.aggressor_side == "buy":
                    resting_key = (trade.seller_id, "sell", trade.price)
                else:
                    resting_key = (trade.buyer_id, "buy", trade.price)

                dq = resting_lookup.get(resting_key)
                if dq:
                    order_info = dq[0]
                    order_info["remaining"] -= fill_qty
                    if order_info["remaining"] <= 1e-12:
                        conn.execute(
                            "UPDATE orders SET remaining = 0, status = 'filled' WHERE id = ?",
                            (order_info["db_id"],),
                        )
                        dq.popleft()
                    else:
                        conn.execute(
                            "UPDATE orders SET remaining = ? WHERE id = ?",
                            (order_info["remaining"], order_info["db_id"]),
                        )

                if fill_qty < trade.quantity:
                    break

            actual_remaining = quantity - total_filled
            resting_order_id = None
            if order_type == "limit" and actual_remaining > 1e-12 and limit_price is not None:
                norm_price = cda._normalize_price(limit_price)
                cur = conn.execute(
                    "INSERT INTO orders "
                    "(market_id, agent_id, side, price, quantity, remaining, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (market_id, agent_id, side, norm_price, quantity, actual_remaining, now),
                )
                resting_order_id = cur.lastrowid

            price_after = store._cda_reference_price(market_id)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        return {
            "trades": persisted_trades,
            "filled_quantity": total_filled,
            "remaining_quantity": actual_remaining,
            "resting_order_id": resting_order_id,
            "price_before": price_before,
            "price_after": price_after,
        }

    def execute_limit_order(
        self, agent_id: int, market_id: int, side: str,
        quantity: float, price: float,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper around execute_cda_order."""
        return self.execute_cda_order(
            market_id, agent_id, side, quantity, price, "limit",
        )

    def execute_market_order(
        self, agent_id: int, market_id: int, side: str, quantity: float,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper around execute_cda_order."""
        return self.execute_cda_order(
            market_id, agent_id, side, quantity, None, "market",
        )

    # ── CDA internal helpers ───────────────────────────────────────────

    @staticmethod
    def _hydrate_cda(store: MarketStore, mkt):
        """
        Reconstruct a ContinuousDoubleAuction from DB order book state.
        Returns (cda, resting_lookup) where resting_lookup maps
        (agent_id, side, price) -> deque of {db_id, remaining} dicts
        for reconciling fills back to DB rows.
        """
        cda = ContinuousDoubleAuction(
            tick_size=mkt["tick_size"],
            min_price=mkt["min_price"],
            max_price=mkt["max_price"],
            initial_reference_price=mkt["initial_price"],
        )
        if mkt["last_trade_price"] is not None:
            cda.last_trade_price = mkt["last_trade_price"]

        resting_lookup: Dict[tuple, deque] = defaultdict(deque)

        orders = store.conn.execute(
            "SELECT * FROM orders WHERE market_id = ? AND status = 'open' ORDER BY id ASC",
            (mkt["id"],),
        ).fetchall()
        for o in orders:
            cda._add_resting_order(
                agent_id=o["agent_id"],
                side=o["side"],
                quantity=float(o["remaining"]),
                price=float(o["price"]),
            )
            key = (o["agent_id"], o["side"], float(o["price"]))
            resting_lookup[key].append({
                "db_id": o["id"],
                "remaining": float(o["remaining"]),
            })

        return cda, resting_lookup
