"""
Autonomous CRRA trader that polls the market API and submits trades.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Any, Dict, List, Mapping, Optional

import requests

try:
    from .crra_math import compute_optimal_trade
except ImportError:
    from crra_math import compute_optimal_trade


DEFAULT_PERSONALITY: Dict[str, float] = {
    "check_interval_mean": 2.0,
    "check_interval_jitter": 1.0,
    "edge_threshold": 0.03,
    "participation_rate": 0.80,
    "trade_size_noise": 0.20,
    "signal_sensitivity": 0.50,
    "stubbornness": 0.30,
}


class AutonomousAgent:
    """Polls the API, computes a CRRA trade, and optionally submits it."""

    def __init__(
        self,
        agent_id,
        api_base_url,
        personality: Optional[Mapping[str, Any]],
        belief,
        rho,
        cash,
        *,
        timeout: float = 5.0,
        rng: Optional[random.Random] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.agent_id = int(agent_id)
        self.api_base_url = str(api_base_url).rstrip("/")
        self.belief = float(belief)
        self.rho = float(rho)
        self.cash = float(cash)
        self.timeout = float(timeout)
        self.rng = rng if rng is not None else random.Random()
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.session = requests.Session()
        self._stop_flag = threading.Event()
        self._shares_by_market: Dict[int, float] = {}

        merged = dict(DEFAULT_PERSONALITY)
        if personality is not None:
            if isinstance(personality, Mapping):
                merged.update(personality)
            else:
                merged.update(vars(personality))
        self.personality = merged

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.api_base_url}{path}"

    def _personality_float(self, key: str) -> float:
        return float(self.personality.get(key, DEFAULT_PERSONALITY[key]))

    def _wait_for_next_cycle(self) -> bool:
        mean = max(0.0, self._personality_float("check_interval_mean"))
        jitter = max(0.0, self._personality_float("check_interval_jitter"))
        sleep_low = max(0.0, mean - jitter)
        sleep_high = max(sleep_low, mean + jitter)
        return self._stop_flag.wait(self.rng.uniform(sleep_low, sleep_high))

    def _wait_after_conflict(self) -> bool:
        return self._stop_flag.wait(1.0)

    def list_open_markets(self) -> List[Dict[str, Any]]:
        response = self.session.get(
            self._url("/markets?status=running"),
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"markets request failed with status {response.status_code}: {response.text}"
            )
        payload = response.json()
        markets = payload.get("markets", payload) if isinstance(payload, dict) else payload
        if not isinstance(markets, list):
            raise RuntimeError("markets response must be a list or {\"markets\": [...]} payload")
        return markets

    def _choose_market(self, markets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not markets:
            return None

        scored: List[tuple[float, Dict[str, Any]]] = []
        for market in markets:
            try:
                price = float(market["price"])
            except (KeyError, TypeError, ValueError):
                continue
            scored.append((abs(self.belief - price), market))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            best_edge = scored[0][0]
            best = [market for edge, market in scored if edge == best_edge]
            if len(best) == 1:
                return best[0]
            idx = min(int(self.rng.uniform(0, len(best))), len(best) - 1)
            return best[idx]

        idx = min(int(self.rng.uniform(0, len(markets))), len(markets) - 1)
        return markets[idx]

    def get_price_snapshot(self, market_id: int) -> Dict[str, Any]:
        response = self.session.get(
            self._url(f"/market/{market_id}/price"),
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"price request failed with status {response.status_code}: {response.text}"
            )
        return response.json()

    def get_agent_state(self, market_id: int) -> Optional[Dict[str, Any]]:
        response = self.session.get(
            self._url(f"/market/{market_id}/agent/{self.agent_id}"),
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise RuntimeError(
                f"agent request failed with status {response.status_code}: {response.text}"
            )
        return response.json()

    def submit_trade(self, market_id: int, quantity: float) -> Optional[Dict[str, Any]]:
        response = self.session.post(
            self._url(f"/market/{market_id}/trade"),
            json={"agent_id": self.agent_id, "quantity": float(quantity)},
            timeout=self.timeout,
        )
        if response.status_code == 409:
            self.logger.warning(
                "Market %s is stopped; agent %s will retry.",
                market_id,
                self.agent_id,
            )
            return None
        if response.status_code >= 400:
            raise RuntimeError(
                f"trade request failed with status {response.status_code}: {response.text}"
            )
        return response.json()

    def run_cycle(self) -> str:
        markets = self.list_open_markets()
        market = self._choose_market(markets)
        if market is None:
            return "no_markets"

        market_id = int(market.get("id", market.get("market_id")))
        price_snapshot = self.get_price_snapshot(market_id)
        agent_state = self.get_agent_state(market_id)

        price = float(price_snapshot["price"])
        if agent_state is None:
            belief = self.belief
            cash = self.cash
            shares = float(self._shares_by_market.get(market_id, 0.0))
            rho = self.rho
        else:
            belief = float(agent_state["belief"])
            cash = float(agent_state["cash"])
            shares = float(agent_state["shares"])
            rho = float(agent_state["rho"])
            self.belief = belief
            self.cash = cash
            self.rho = rho
            self._shares_by_market[market_id] = shares

        x_star = compute_optimal_trade(
            belief=belief,
            price=price,
            cash=cash,
            shares=shares,
            rho=rho,
        )

        trade_size_noise = min(max(self._personality_float("trade_size_noise"), 0.0), 1.0)
        x_star *= self.rng.uniform(1.0 - trade_size_noise, 1.0 + trade_size_noise)

        edge_threshold = max(0.0, self._personality_float("edge_threshold"))
        if abs(belief - price) < edge_threshold:
            return "edge_too_small"

        participation_rate = min(max(self._personality_float("participation_rate"), 0.0), 1.0)
        if self.rng.random() > participation_rate:
            return "skipped_participation"

        if abs(x_star) < 1e-9:
            return "trade_too_small"

        trade_result = self.submit_trade(market_id, x_star)
        if trade_result is None:
            return "retry"

        if "agent_cash_after" in trade_result:
            self.cash = float(trade_result["agent_cash_after"])
        if "agent_shares_after" in trade_result:
            self._shares_by_market[market_id] = float(trade_result["agent_shares_after"])

        self.logger.info(
            "Agent %s traded %.6f shares in market %s.",
            self.agent_id,
            x_star,
            market_id,
        )
        return "traded"

    def stop(self):
        self._stop_flag.set()

    def run(self):
        while not self._stop_flag.is_set():
            if self._wait_for_next_cycle():
                break
            try:
                outcome = self.run_cycle()
                if outcome == "retry" and self._wait_after_conflict():
                    break
            except Exception:
                self.logger.exception(
                    "Autonomous agent %s failed during market cycle.",
                    self.agent_id,
                )
