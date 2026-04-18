"""
Multi-market autonomous agent orchestrator.

Owns agent worker threads globally (per agent, not per market) and manages
market lifecycle transitions used by API start/stop endpoints.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

from autonomous_agent import AutonomousAgent
from market_service import MarketService

logger = logging.getLogger(__name__)


@dataclass
class _AgentSeed:
    agent_id: int
    belief: float
    rho: float
    cash: float
    personality: Optional[Dict[str, Any]]


@dataclass
class _AgentState:
    agent: Any
    thread: threading.Thread
    markets: Set[int] = field(default_factory=set)
    restart_count: int = 0
    stop_requested: bool = False


class AgentRunner:
    """
    Global autonomous runner across markets.

    A single agent thread can be attached to multiple running markets.
    The agent chooses where to trade each cycle via market discovery.
    """

    def __init__(
        self,
        *,
        api_base_url: str,
        market_service: MarketService,
        max_restarts: int = 3,
        monitor_interval_sec: float = 1.0,
        agent_factory: Optional[Callable[..., Any]] = None,
    ):
        self._api_base_url = api_base_url.rstrip("/")
        self._market_service = market_service
        self._max_restarts = int(max_restarts)
        self._monitor_interval_sec = float(max(0.1, monitor_interval_sec))
        self._agent_factory = agent_factory or AutonomousAgent

        self._lock = threading.RLock()
        self._market_agents: Dict[int, Set[int]] = {}
        self._market_started_at: Dict[int, float] = {}
        self._agent_seeds: Dict[int, _AgentSeed] = {}
        self._agent_states: Dict[int, _AgentState] = {}

        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def start_market(self, market_id: int) -> int:
        """
        Mark market running and attach required agent threads.

        Returns number of active agent threads for this market.
        """
        with self._lock:
            if market_id in self._market_agents:
                raise ValueError(f"Market {market_id} already running")

        self._market_service.get_market(market_id)
        self._market_service.set_market_status(market_id, "running")
        rows = self._market_service.list_agents(limit=1_000_000, offset=0)["agents"]

        with self._lock:
            self._ensure_monitor_locked()
            started_agent_ids: Set[int] = set()
            for row in rows:
                seed = self._seed_from_row(row)
                aid = seed.agent_id
                started_agent_ids.add(aid)
                state = self._agent_states.get(aid)
                if state is None:
                    self._start_agent_locked(seed, {market_id})
                else:
                    state.markets.add(market_id)
                    state.stop_requested = False
                self._agent_seeds[aid] = seed
            self._market_agents[market_id] = started_agent_ids
            self._market_started_at[market_id] = time.monotonic()
            return self.agent_count_active(market_id)

    def stop_market(self, market_id: int) -> Dict[str, Any]:
        """
        Mark market stopped and stop agents no longer used by other markets.
        """
        join_targets = []
        with self._lock:
            if market_id not in self._market_agents:
                raise ValueError(f"Market {market_id} is not running")
            agent_ids = self._market_agents.pop(market_id)
            started_at = self._market_started_at.pop(market_id, time.monotonic())

            for aid in agent_ids:
                state = self._agent_states.get(aid)
                if state is None:
                    continue
                state.markets.discard(market_id)
                if state.markets:
                    continue
                state.stop_requested = True
                try:
                    state.agent.stop()
                except Exception:
                    logger.exception("Agent %s stop() failed", aid)
                join_targets.append((aid, state.thread))

        self._market_service.set_market_status(market_id, "stopped")
        zombies = self._join_and_prune(join_targets)

        with self._lock:
            remaining = len(self._market_agents)
            if remaining == 0:
                self._request_monitor_shutdown_locked()

        duration = max(0.0, time.monotonic() - started_at)
        return {
            "market_id": market_id,
            "duration_sec": duration,
            "agents_detached": len(agent_ids),
            "zombie_threads": zombies,
            "markets_running": remaining,
        }

    def start_all(self) -> Dict[int, int]:
        """
        Start autonomous trading for every market in status=open.
        """
        out: Dict[int, int] = {}
        for m in self._market_service.list_markets(status="open"):
            mid = int(m["id"])
            if self.is_running(mid):
                out[mid] = self.agent_count_active(mid)
                continue
            out[mid] = self.start_market(mid)
        return out

    def stop_all(self) -> Dict[int, Dict[str, Any]]:
        """
        Stop autonomous trading for all currently running markets.
        """
        with self._lock:
            market_ids = list(self._market_agents.keys())
        out: Dict[int, Dict[str, Any]] = {}
        for mid in market_ids:
            try:
                out[mid] = self.stop_market(mid)
            except ValueError:
                continue
        return out

    def is_running(self, market_id: int) -> bool:
        with self._lock:
            return market_id in self._market_agents

    def agent_count_active(self, market_id: int) -> int:
        with self._lock:
            aids = self._market_agents.get(market_id, set())
            return sum(
                1
                for aid in aids
                if aid in self._agent_states and self._agent_states[aid].thread.is_alive()
            )

    def shutdown(self) -> None:
        """
        Full runner shutdown for test/runtime reset.
        """
        self.stop_all()
        self._request_monitor_shutdown()

    def register_or_update_agent(self, agent: Dict[str, Any]) -> None:
        """
        Register agent metadata and attach thread to all running markets.

        Called by API agent create/update endpoints so autonomous execution stays
        global and independent from per-market joins.
        """
        seed = self._seed_from_row(agent)
        aid = seed.agent_id
        with self._lock:
            self._agent_seeds[aid] = seed
            running_markets = set(self._market_agents.keys())
            if not running_markets:
                return
            self._ensure_monitor_locked()
            state = self._agent_states.get(aid)
            if state is None:
                self._start_agent_locked(seed, running_markets)
            else:
                state.markets.update(running_markets)
                state.stop_requested = False
            for mid in running_markets:
                self._market_agents[mid].add(aid)

    def _seed_from_row(self, row: Dict[str, Any]) -> _AgentSeed:
        raw_personality = row.get("personality")
        personality: Optional[Dict[str, Any]] = None
        if isinstance(raw_personality, dict):
            personality = raw_personality
        elif isinstance(raw_personality, str) and raw_personality.strip():
            try:
                parsed = json.loads(raw_personality)
                if isinstance(parsed, dict):
                    personality = parsed
            except json.JSONDecodeError:
                personality = None
        return _AgentSeed(
            agent_id=int(row.get("agent_id", row.get("id"))),
            belief=float(row.get("belief") or 0.5),
            rho=float(row.get("rho") or 1.0),
            cash=float(row.get("cash") or 0.0),
            personality=personality,
        )

    def _start_agent_locked(self, seed: _AgentSeed, markets: Set[int]) -> None:
        agent = self._agent_factory(
            agent_id=seed.agent_id,
            api_base_url=self._api_base_url,
            personality=seed.personality,
            belief=seed.belief,
            rho=seed.rho,
            cash=seed.cash,
        )
        thread = threading.Thread(
            target=self._run_agent_loop,
            args=(seed.agent_id, agent),
            name=f"agent-runner-agent-{seed.agent_id}",
            daemon=True,
        )
        self._agent_states[seed.agent_id] = _AgentState(
            agent=agent,
            thread=thread,
            markets=set(markets),
        )
        thread.start()

    def _run_agent_loop(self, agent_id: int, agent: Any) -> None:
        try:
            agent.run()
        except Exception:
            logger.exception("Autonomous agent thread crashed for agent %s", agent_id)

    def _ensure_monitor_locked(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="agent-runner-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _request_monitor_shutdown_locked(self) -> None:
        self._monitor_stop.set()
        self._monitor_thread = None

    def _request_monitor_shutdown(self) -> None:
        monitor = None
        with self._lock:
            monitor = self._monitor_thread
            self._request_monitor_shutdown_locked()
        if monitor is not None and monitor.is_alive():
            monitor.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self._monitor_interval_sec):
            with self._lock:
                for aid, state in list(self._agent_states.items()):
                    if state.stop_requested:
                        continue
                    if state.thread.is_alive():
                        continue
                    if not state.markets:
                        self._agent_states.pop(aid, None)
                        self._agent_seeds.pop(aid, None)
                        continue
                    if state.restart_count >= self._max_restarts:
                        logger.error(
                            "Agent %s exceeded restart cap (%s); keeping stopped",
                            aid,
                            self._max_restarts,
                        )
                        state.stop_requested = True
                        continue
                    seed = self._agent_seeds.get(aid)
                    if seed is None:
                        try:
                            full = self._market_service.get_agent(aid)
                        except ValueError:
                            self._agent_states.pop(aid, None)
                            continue
                        seed = _AgentSeed(
                            agent_id=aid,
                            belief=float(full.get("belief") or 0.5),
                            rho=float(full.get("rho") or 1.0),
                            cash=float(full.get("cash") or 0.0),
                            personality=None,
                        )
                        self._agent_seeds[aid] = seed
                    state.restart_count += 1
                    logger.warning(
                        "Restarting autonomous agent %s (%s/%s)",
                        aid,
                        state.restart_count,
                        self._max_restarts,
                    )
                    agent = self._agent_factory(
                        agent_id=seed.agent_id,
                        api_base_url=self._api_base_url,
                        personality=seed.personality,
                        belief=seed.belief,
                        rho=seed.rho,
                        cash=seed.cash,
                    )
                    thread = threading.Thread(
                        target=self._run_agent_loop,
                        args=(aid, agent),
                        name=f"agent-runner-agent-{aid}",
                        daemon=True,
                    )
                    state.agent = agent
                    state.thread = thread
                    thread.start()

    def _join_and_prune(self, join_targets) -> int:
        zombies = 0
        for _aid, thread in join_targets:
            thread.join(timeout=5.0)

        with self._lock:
            for aid, original_thread in join_targets:
                state = self._agent_states.get(aid)
                if state is None:
                    continue
                if state.markets:
                    continue
                if state.thread is not original_thread:
                    continue
                if state.thread.is_alive():
                    zombies += 1
                    continue
                self._agent_states.pop(aid, None)
                self._agent_seeds.pop(aid, None)
        return zombies
