from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from typing import Dict, Set

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "app"), os.path.join(ROOT, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent_runner import AgentRunner
from market_service import MarketService


class FakeAutonomousAgent:
    crash_once_agents: Set[int] = set()
    created_by_agent: Dict[int, int] = {}

    def __init__(self, agent_id, api_base_url, personality, belief, rho, cash, **kwargs):
        self.agent_id = int(agent_id)
        self._stop = threading.Event()
        FakeAutonomousAgent.created_by_agent[self.agent_id] = (
            FakeAutonomousAgent.created_by_agent.get(self.agent_id, 0) + 1
        )

    def run(self):
        if self.agent_id in FakeAutonomousAgent.crash_once_agents:
            FakeAutonomousAgent.crash_once_agents.remove(self.agent_id)
            raise RuntimeError("planned crash")
        while not self._stop.wait(0.01):
            pass

    def stop(self):
        self._stop.set()


@pytest.fixture
def svc():
    base = os.path.join(ROOT, "tmp_runner_tests", str(uuid.uuid4()))
    os.makedirs(base, exist_ok=True)
    s = MarketService(os.path.join(base, "runner.db"))
    yield s
    s.close()
    shutil.rmtree(base, ignore_errors=True)


def _seed_market(svc: MarketService, slug: str):
    mkt = svc.create_market(slug=slug, title=slug, mechanism="lmsr", b=100.0)
    svc.set_market_status(mkt["id"], "open")
    return int(mkt["id"])


def _seed_agents(svc: MarketService, slug: str, n_agents: int):
    agent_ids = []
    for i in range(n_agents):
        a = svc.create_agent(
            name=f"{slug}-a{i}",
            cash=500.0,
            belief=0.6,
            rho=1.0,
            personality="{}",
        )
        agent_ids.append(int(a["id"]))
    return agent_ids


def test_start_stop_two_markets_no_zombie_threads(svc: MarketService):
    FakeAutonomousAgent.created_by_agent.clear()
    FakeAutonomousAgent.crash_once_agents.clear()
    _seed_agents(svc, "base", 10)
    m1 = _seed_market(svc, "m1")
    m2 = _seed_market(svc, "m2")
    runner = AgentRunner(
        api_base_url="http://127.0.0.1:8000/api",
        market_service=svc,
        monitor_interval_sec=0.05,
        agent_factory=FakeAutonomousAgent,
    )

    assert runner.start_market(m1) == 10
    assert runner.start_market(m2) == 10
    assert runner.is_running(m1)
    assert runner.is_running(m2)

    stop1 = runner.stop_market(m1)
    assert stop1["zombie_threads"] == 0
    assert not runner.is_running(m1)
    assert runner.is_running(m2)
    assert runner.agent_count_active(m2) == 10

    stop2 = runner.stop_market(m2)
    assert stop2["zombie_threads"] == 0
    assert not runner.is_running(m2)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        leaked = [
            t for t in threading.enumerate() if t.name.startswith("agent-runner-")
        ]
        if not leaked:
            break
        time.sleep(0.05)
    assert [
        t.name for t in threading.enumerate() if t.name.startswith("agent-runner-")
    ] == []

    runner.shutdown()


def test_restarts_dead_agent_thread(svc: MarketService):
    FakeAutonomousAgent.created_by_agent.clear()
    agent_ids = _seed_agents(svc, "crash", 1)
    m1 = _seed_market(svc, "crash-mkt")
    crashing_agent_id = agent_ids[0]
    FakeAutonomousAgent.crash_once_agents = {crashing_agent_id}

    runner = AgentRunner(
        api_base_url="http://127.0.0.1:8000/api",
        market_service=svc,
        monitor_interval_sec=0.05,
        agent_factory=FakeAutonomousAgent,
    )
    runner.start_market(m1)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if FakeAutonomousAgent.created_by_agent.get(crashing_agent_id, 0) >= 2:
            break
        time.sleep(0.05)

    assert FakeAutonomousAgent.created_by_agent.get(crashing_agent_id, 0) >= 2
    assert runner.agent_count_active(m1) == 1
    runner.stop_market(m1)
    runner.shutdown()
