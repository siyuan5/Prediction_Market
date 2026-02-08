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
    aggressor_side: str


@dataclass
class RestingOrder:
    order_id: int
    agent_id: int
    side: str
    price: float
    remaining: float
    timestamp: int


class ContinuousDoubleAuction:
    """
    Continuous Double Auction exchange for a binary claim.
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

        # Price -> FIFO queue of resting orders at that level.
        self._bid_levels: Dict[float, Deque[RestingOrder]] = {}
        self._ask_levels: Dict[float, Deque[RestingOrder]] = {}
        # Sorted price ladders for fast best bid/ask lookup.
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []

        self._next_order_id = 1
        self._clock = 0
        self.last_trade_price: Optional[float] = None
        self._fallback_price = float(initial_reference_price)

    def _normalize_price(self, price: float) -> float:
        # Exchange accepts only bounded, tick-aligned prices.
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
        order = RestingOrder(
            order_id=self._next_order_id,
            agent_id=agent_id,
            side=side,
            price=price,
            remaining=quantity,
            timestamp=self._clock,
        )
        self._next_order_id += 1

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
        if not self._bid_prices:
            return None
        return self._bid_prices[-1]

    def best_ask(self) -> Optional[float]:
        if not self._ask_prices:
            return None
        return self._ask_prices[0]

    def mid_price(self) -> Optional[float]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def reference_price(self) -> float:
        # Mid quote if available, otherwise last trade, then top-of-book fallback.
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
        self._cancel_agent_orders_on_side(agent_id=agent_id, side="buy")
        self._cancel_agent_orders_on_side(agent_id=agent_id, side="sell")

    def _cancel_agent_orders_on_side(self, *, agent_id: int, side: str) -> None:
        if side == "buy":
            prices = list(self._bid_prices)
            levels = self._bid_levels
        else:
            prices = list(self._ask_prices)
            levels = self._ask_levels

        for price in prices:
            level = levels[price]
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
        normalized_price = self._normalize_price(limit_price)
        return self._submit_order(
            agent_id=agent_id,
            side=side,
            quantity=quantity,
            limit_price=normalized_price,
            is_market=False,
        )

    def submit_market_order(self, *, agent_id: int, side: str, quantity: float) -> dict:
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
            return {
                "trades": [],
                "filled_quantity": 0.0,
                "remaining_quantity": 0.0,
                "resting_order_id": None,
            }

        self._clock += 1
        trades: List[Trade] = []
        eps = 1e-12

        # Aggressively match against the opposite side until price/size constraints stop us.
        while remaining > eps:
            if side == "buy":
                best_ask = self.best_ask()
                if best_ask is None:
                    break
                if (not is_market) and (limit_price is not None) and (limit_price < best_ask):
                    break

                ask_queue = self._ask_levels[best_ask]
                # Price-time priority: oldest order at best price fills first.
                resting = ask_queue[0]
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
                if (not is_market) and (limit_price is not None) and (limit_price > best_bid):
                    break

                bid_queue = self._bid_levels[best_bid]
                # Price-time priority: oldest order at best price fills first.
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
        # Unfilled remainder of a limit order becomes new resting liquidity.
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
