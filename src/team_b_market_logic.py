import bisect
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class Trade:
    buyer_id: int
    seller_id: int
    price: float
    quantity: float
    aggressor_side: str  # 'buy' or 'sell' (side of order that crossed the spread)


@dataclass
class RestingOrder:
    order_id: int
    agent_id: int
    side: str          # 'buy' or 'sell'
    price: float
    remaining: float   # shares remaining on this order
    timestamp: int     # submission time for FIFO priority


class ContinuousDoubleAuction:
    """
    Implementation of a Continuous Double Auction (CDA) for a binary market.
    Maintains full visible limit order book (no hidden orders).
    """

    def __init__(
        self,
        *,
        tick_size: float = 1e-4,
        min_price: float = 0.001,
        max_price: float = 0.999,
        initial_reference_price: float = 0.5,
    ):
        self.tick_size = float(tick_size)
        self.min_price = float(min_price)
        self.max_price = float(max_price)

        # Mapping price -> FIFO resting orders on each side of the book
        self._bid_levels: Dict[float, Deque[RestingOrder]] = {}
        self._ask_levels: Dict[float, Deque[RestingOrder]] = {}
        # Sorted lists for best bid/ask lookups (descending for bids, ascending for asks)
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []

        self._next_order_id = 1        # Assign unique IDs to all resting orders
        self._clock = 0                # Logical clock for FIFO/tiebreaks
        self.last_trade_price: Optional[float] = None
        self._fallback_price = float(initial_reference_price)  # Used if no quotes/trades yet

    def _normalize_price(self, price: float) -> float:
        # Clip price into market bounds and align it to tick size
        clipped = min(self.max_price, max(self.min_price, float(price)))
        ticks = round(clipped / self.tick_size)
        return ticks * self.tick_size

    def _add_resting_order(
        self,
        *,
        agent_id: int,
        side: str,
        quantity: float,
        price: float,
    ) -> int:
        # Store a new resting order; ensure correct book structure and price ladder maintenance
        order = RestingOrder(
            order_id=self._next_order_id,
            agent_id=agent_id,
            side=side,
            price=price,
            remaining=quantity,
            timestamp=self._clock,
        )
        self._next_order_id += 1

        # Add to bid/ask level, maintaining sorted price ladder for fast best bid/ask queries
        if side == "buy":
            if price not in self._bid_levels:
                self._bid_levels[price] = deque()
                bisect.insort(self._bid_prices, price)
            self._bid_levels[price].append(order)
        else:
            if price not in self._ask_levels:
                self._ask_levels[price] = deque()
                bisect.insort(self._ask_prices, price)
            self._ask_levels[price].append(order)

        return order.order_id

    def _remove_price_level_if_empty(self, *, side: str, price: float) -> None:
        # Remove price level from book and price ladder if no orders remain there
        if side == "buy":
            level = self._bid_levels.get(price)
            if level is not None and len(level) == 0:
                del self._bid_levels[price]
                idx = bisect.bisect_left(self._bid_prices, price)
                if idx < len(self._bid_prices) and self._bid_prices[idx] == price:
                    self._bid_prices.pop(idx)
        else:
            level = self._ask_levels.get(price)
            if level is not None and len(level) == 0:
                del self._ask_levels[price]
                idx = bisect.bisect_left(self._ask_prices, price)
                if idx < len(self._ask_prices) and self._ask_prices[idx] == price:
                    self._ask_prices.pop(idx)

    def best_bid(self) -> Optional[float]:
        # Highest bid price available, None if no bids
        if not self._bid_prices:
            return None
        return self._bid_prices[-1]

    def best_ask(self) -> Optional[float]:
        # Lowest ask price available, None if no asks
        if not self._ask_prices:
            return None
        return self._ask_prices[0]

    def mid_price(self) -> Optional[float]:
        # Return mid-point of best bid/ask or None if either book empty
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def reference_price(self) -> float:
        # Midpoint if available, then last trade, then best bid/ask, then fallback
        mid = self.mid_price()
        if mid is not None:
            return mid
        if self.last_trade_price is not None:
            return self.last_trade_price
        bid = self.best_bid()
        if bid is not None:
            return bid
        ask = self.best_ask()
        if ask is not None:
            return ask
        return self._fallback_price

    def cancel_agent_orders(self, agent_id: int) -> None:
        # Remove all resting orders submitted by this agent from the book
        self._cancel_agent_orders_on_side(agent_id=agent_id, side="buy")
        self._cancel_agent_orders_on_side(agent_id=agent_id, side="sell")

    def _cancel_agent_orders_on_side(self, *, agent_id: int, side: str) -> None:
        # Cancel only on a specific side of the book
        if side == "buy":
            prices = list(self._bid_prices)
            levels = self._bid_levels
        else:
            prices = list(self._ask_prices)
            levels = self._ask_levels

        for price in prices:
            level = levels[price]
            # Filter out all orders from the given agent, retain rest
            filtered = deque(order for order in level if order.agent_id != agent_id)
            levels[price] = filtered
            self._remove_price_level_if_empty(side=side, price=price)

    def submit_limit_order(
        self,
        *,
        agent_id: int,
        side: str,
        quantity: float,
        limit_price: float,
    ) -> dict:
        # Price-improving orders may match immediately; remainder adds liquidity to book
        normalized_price = self._normalize_price(limit_price)
        return self._submit_order(
            agent_id=agent_id,
            side=side,
            quantity=quantity,
            limit_price=normalized_price,
            is_market=False,
        )

    def submit_market_order(self, *, agent_id: int, side: str, quantity: float) -> dict:
        # Aggressively cross the spread at any price until quantity filled
        return self._submit_order(
            agent_id=agent_id,
            side=side,
            quantity=quantity,
            limit_price=None,
            is_market=True,
        )

    def _submit_order(
        self,
        *,
        agent_id: int,
        side: str,
        quantity: float,
        limit_price: Optional[float],
        is_market: bool,
    ) -> dict:
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")

        remaining = float(quantity)
        if remaining <= 0:
            # Skip empty or negative orders (should not happen in simulation)
            return {
                "trades": [],
                "filled_quantity": 0.0,
                "remaining_quantity": 0.0,
                "resting_order_id": None,
            }

        self._clock += 1
        trades: List[Trade] = []
        eps = 1e-12  # Tolerance for floating point comparison

        # Main matching loop: fill incoming order against top of book, in price-time order
        while remaining > eps:
            if side == "buy":
                best_ask = self.best_ask()
                if best_ask is None:
                    break  # Nothing to match
                # For limit: ensure we don't pay more than limit price
                if (not is_market) and (limit_price is not None) and (limit_price < best_ask):
                    break

                ask_queue = self._ask_levels[best_ask]
                resting = ask_queue[0]  # FIFO: oldest order at best price
                executed = min(remaining, resting.remaining)
                trade_price = resting.price
                trades.append(
                    Trade(
                        buyer_id=agent_id,
                        seller_id=resting.agent_id,
                        price=trade_price,
                        quantity=executed,
                        aggressor_side="buy",
                    )
                )
                remaining -= executed
                resting.remaining -= executed
                self.last_trade_price = trade_price

                if resting.remaining <= eps:
                    ask_queue.popleft()
                    self._remove_price_level_if_empty(side="sell", price=best_ask)
            else:
                best_bid = self.best_bid()
                if best_bid is None:
                    break
                # For limit: ensure we don't sell at a price lower than limit
                if (not is_market) and (limit_price is not None) and (limit_price > best_bid):
                    break

                bid_queue = self._bid_levels[best_bid]
                resting = bid_queue[0]
                executed = min(remaining, resting.remaining)
                trade_price = resting.price
                trades.append(
                    Trade(
                        buyer_id=resting.agent_id,
                        seller_id=agent_id,
                        price=trade_price,
                        quantity=executed,
                        aggressor_side="sell",
                    )
                )
                remaining -= executed
                resting.remaining -= executed
                self.last_trade_price = trade_price

                if resting.remaining <= eps:
                    bid_queue.popleft()
                    self._remove_price_level_if_empty(side="buy", price=best_bid)

        resting_order_id = None
        # For limit orders, any unfilled quantity gets posted to the book as new liquidity
        if (not is_market) and remaining > eps and limit_price is not None:
            resting_order_id = self._add_resting_order(
                agent_id=agent_id,
                side=side,
                quantity=remaining,
                price=limit_price,
            )

        return {
            "trades": trades,
            "filled_quantity": quantity - remaining,
            "remaining_quantity": remaining,
            "resting_order_id": resting_order_id,
        }
