from autonomous_agent import AutonomousAgent


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.text = text or str(self._payload)

    def json(self):
        return self._payload


class FakeRng:
    def __init__(self, *, uniform_values=None, random_values=None):
        self.uniform_values = list(uniform_values or [1.0])
        self.random_values = list(random_values or [0.0])

    def uniform(self, _a, _b):
        if self.uniform_values:
            return self.uniform_values.pop(0)
        return 1.0

    def random(self):
        if self.random_values:
            return self.random_values.pop(0)
        return 0.0


def test_run_cycle_submits_trade_when_edge_is_large():
    agent = AutonomousAgent(
        agent_id=7,
        api_base_url="http://127.0.0.1:8000/api",
        belief=0.80,
        rho=1.0,
        cash=100.0,
        personality={
            "edge_threshold": 0.03,
            "participation_rate": 1.0,
            "trade_size_noise": 0.0,
        },
        rng=FakeRng(uniform_values=[1.0], random_values=[0.0]),
    )

    submitted = {}
    responses = [
        FakeResponse(
            payload={
                "markets": [
                    {"id": 1, "price": 0.40},
                    {"id": 2, "price": 0.72},
                ]
            }
        ),
        FakeResponse(payload={"price": 0.40}),
        FakeResponse(payload={"cash": 100.0, "shares": 0.0, "belief": 0.80, "rho": 1.0}),
    ]

    def fake_get(_url, timeout):
        assert timeout == 5.0
        return responses.pop(0)

    def fake_post(_url, json, timeout):
        assert timeout == 5.0
        submitted.update(json)
        return FakeResponse(payload={"trade_id": "t1"})

    agent.session.get = fake_get
    agent.session.post = fake_post

    outcome = agent.run_cycle()

    assert outcome == "traded"
    assert submitted["agent_id"] == 7
    assert submitted["quantity"] > 0.0
    assert agent._shares_by_market[1] >= 0.0


def test_run_cycle_skips_when_edge_is_below_threshold():
    agent = AutonomousAgent(
        agent_id=2,
        api_base_url="http://127.0.0.1:8000/api",
        belief=0.52,
        rho=1.0,
        cash=100.0,
        personality={
            "edge_threshold": 0.03,
            "participation_rate": 1.0,
            "trade_size_noise": 0.0,
        },
        rng=FakeRng(uniform_values=[1.0], random_values=[0.0]),
    )

    responses = [
        FakeResponse(payload={"markets": [{"id": 1, "price": 0.50}]}),
        FakeResponse(payload={"price": 0.50}),
        FakeResponse(payload={"cash": 100.0, "shares": 0.0, "belief": 0.52, "rho": 1.0}),
    ]
    post_calls = []

    agent.session.get = lambda _url, timeout: responses.pop(0)
    agent.session.post = lambda _url, json, timeout: post_calls.append(json)

    outcome = agent.run_cycle()

    assert outcome == "edge_too_small"
    assert post_calls == []


def test_run_cycle_skips_when_participation_check_fails():
    agent = AutonomousAgent(
        agent_id=3,
        api_base_url="http://127.0.0.1:8000/api",
        belief=0.80,
        rho=1.0,
        cash=100.0,
        personality={
            "edge_threshold": 0.01,
            "participation_rate": 0.20,
            "trade_size_noise": 0.0,
        },
        rng=FakeRng(uniform_values=[1.0], random_values=[0.95]),
    )

    responses = [
        FakeResponse(payload={"markets": [{"id": 1, "price": 0.40}]}),
        FakeResponse(payload={"price": 0.40}),
        FakeResponse(payload={"cash": 100.0, "shares": 0.0, "belief": 0.80, "rho": 1.0}),
    ]
    post_calls = []

    agent.session.get = lambda _url, timeout: responses.pop(0)
    agent.session.post = lambda _url, json, timeout: post_calls.append(json)

    outcome = agent.run_cycle()

    assert outcome == "skipped_participation"
    assert post_calls == []


def test_run_cycle_uses_constructor_state_before_position_exists():
    agent = AutonomousAgent(
        agent_id=5,
        api_base_url="http://127.0.0.1:8000/api",
        belief=0.80,
        rho=1.0,
        cash=250.0,
        personality={
            "edge_threshold": 0.01,
            "participation_rate": 1.0,
            "trade_size_noise": 0.0,
        },
        rng=FakeRng(uniform_values=[1.0], random_values=[0.0]),
    )

    get_calls = []

    def fake_get(_url, timeout):
        get_calls.append(_url)
        if "/markets" in _url:
            return FakeResponse(payload={"markets": [{"id": 11, "price": 0.30}]})
        if _url.endswith("/price"):
            return FakeResponse(payload={"price": 0.30})
        return FakeResponse(status_code=404, payload={"detail": "agent has no position"})

    submitted = {}

    def fake_post(_url, json, timeout):
        submitted.update(json)
        return FakeResponse(
            payload={
                "trade_id": "t-new",
                "agent_cash_after": 200.0,
                "agent_shares_after": 50.0,
            }
        )

    agent.session.get = fake_get
    agent.session.post = fake_post

    outcome = agent.run_cycle()

    assert outcome == "traded"
    assert submitted["quantity"] > 0.0
    assert agent.cash == 200.0
    assert agent._shares_by_market[11] == 50.0
    assert any(url.endswith("/agent/5") for url in get_calls)


def test_run_retries_after_409_conflict():
    agent = AutonomousAgent(
        agent_id=9,
        api_base_url="http://127.0.0.1:8000/api",
        belief=0.80,
        rho=1.0,
        cash=100.0,
        personality={
            "check_interval_mean": 0.0,
            "check_interval_jitter": 0.0,
            "edge_threshold": 0.01,
            "participation_rate": 1.0,
            "trade_size_noise": 0.0,
        },
        rng=FakeRng(uniform_values=[1.0, 1.0], random_values=[0.0, 0.0]),
    )

    wait_calls = []
    post_statuses = [409, 200]

    def fake_wait_for_next_cycle():
        return False

    def fake_wait_after_conflict():
        wait_calls.append("retry")
        return False

    def fake_get(_url, timeout):
        if "/markets" in _url:
            return FakeResponse(payload={"markets": [{"id": 1, "price": 0.40}]})
        return FakeResponse(
            payload={"price": 0.40}
            if _url.endswith("/price")
            else {"cash": 100.0, "shares": 0.0, "belief": 0.80, "rho": 1.0}
        )

    def fake_post(_url, json, timeout):
        status = post_statuses.pop(0)
        if status == 200:
            agent.stop()
            return FakeResponse(status_code=200, payload={"trade_id": "t2"})
        return FakeResponse(status_code=409, payload={"error": "stopped"})

    agent._wait_for_next_cycle = fake_wait_for_next_cycle
    agent._wait_after_conflict = fake_wait_after_conflict
    agent.session.get = fake_get
    agent.session.post = fake_post

    agent.run()

    assert wait_calls == ["retry"]
    assert post_statuses == []
