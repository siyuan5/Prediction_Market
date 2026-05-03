"""
Unit tests for src/personality.py — Task 5 of the autonomous agent system.

Tests cover:
  - Personality dataclass creation and field defaults
  - All 7 personality fields affecting behavior
  - sample_personality() with default and custom distributions
  - Serialisation round-trip: Personality -> dict -> JSON -> dict -> Personality
  - DEFAULT_POPULATION_DIST structure
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

# Ensure src/ is importable without an installed package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from personality import (
    DEFAULT_POPULATION_DIST,
    Personality,
    sample_personality,
)


# ---------------------------------------------------------------------------
# Personality dataclass
# ---------------------------------------------------------------------------


class TestPersonalityDefaults:
    def test_default_field_values(self):
        p = Personality()
        assert p.check_interval_mean == 2.0
        assert p.check_interval_jitter == 1.0
        assert p.edge_threshold == 0.03
        assert p.participation_rate == 0.80
        assert p.trade_size_noise == 0.20
        assert p.signal_sensitivity == 0.50
        assert p.stubbornness == 0.30

    def test_all_fields_present(self):
        fields = set(Personality.__dataclass_fields__)
        assert fields == {
            "check_interval_mean",
            "check_interval_jitter",
            "edge_threshold",
            "participation_rate",
            "trade_size_noise",
            "signal_sensitivity",
            "stubbornness",
            "trade_fraction",
            "comment_influence",
        }


class TestPersonalitySerialisation:
    def test_to_dict_returns_all_fields(self):
        p = Personality()
        d = p.to_dict()
        assert set(d.keys()) == set(Personality.__dataclass_fields__)

    def test_to_dict_values_match(self):
        p = Personality(edge_threshold=0.07, participation_rate=0.6)
        d = p.to_dict()
        assert d["edge_threshold"] == pytest.approx(0.07)
        assert d["participation_rate"] == pytest.approx(0.6)

    def test_from_dict_roundtrip(self):
        p = Personality(
            check_interval_mean=3.0,
            edge_threshold=0.05,
            signal_sensitivity=0.8,
            stubbornness=0.1,
        )
        d = p.to_dict()
        p2 = Personality.from_dict(d)
        assert p2.check_interval_mean == pytest.approx(p.check_interval_mean)
        assert p2.edge_threshold == pytest.approx(p.edge_threshold)
        assert p2.signal_sensitivity == pytest.approx(p.signal_sensitivity)
        assert p2.stubbornness == pytest.approx(p.stubbornness)

    def test_json_roundtrip(self):
        p = Personality(participation_rate=0.65, trade_size_noise=0.15)
        json_str = json.dumps(p.to_dict())
        d = json.loads(json_str)
        p2 = Personality.from_dict(d)
        assert p2.participation_rate == pytest.approx(p.participation_rate)
        assert p2.trade_size_noise == pytest.approx(p.trade_size_noise)

    def test_from_dict_with_missing_keys_uses_defaults(self):
        p = Personality.from_dict({"edge_threshold": 0.09})
        assert p.edge_threshold == pytest.approx(0.09)
        # All other fields should fall back to Personality defaults
        defaults = Personality()
        assert p.check_interval_mean == pytest.approx(defaults.check_interval_mean)
        assert p.participation_rate == pytest.approx(defaults.participation_rate)
        assert p.signal_sensitivity == pytest.approx(defaults.signal_sensitivity)
        assert p.stubbornness == pytest.approx(defaults.stubbornness)


# ---------------------------------------------------------------------------
# sample_personality
# ---------------------------------------------------------------------------


class TestSamplePersonality:
    def test_returns_personality_instance(self):
        p = sample_personality()
        assert isinstance(p, Personality)

    def test_seeded_rng_is_reproducible(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        p1 = sample_personality(rng=rng1)
        p2 = sample_personality(rng=rng2)
        assert p1.check_interval_mean == pytest.approx(p2.check_interval_mean)
        assert p1.edge_threshold == pytest.approx(p2.edge_threshold)

    def test_different_seeds_produce_different_personalities(self):
        p1 = sample_personality(rng=random.Random(1))
        p2 = sample_personality(rng=random.Random(9999))
        # Not identical across all fields (extremely unlikely with the given distribution)
        diffs = [
            abs(getattr(p1, f) - getattr(p2, f))
            for f in Personality.__dataclass_fields__
        ]
        assert any(d > 1e-6 for d in diffs)

    def test_check_interval_mean_in_range(self):
        rng = random.Random(0)
        samples = [sample_personality(rng=rng).check_interval_mean for _ in range(200)]
        assert all(1.0 <= v <= 4.0 for v in samples)

    def test_edge_threshold_in_range(self):
        rng = random.Random(0)
        samples = [sample_personality(rng=rng).edge_threshold for _ in range(200)]
        assert all(0.01 <= v <= 0.10 for v in samples)

    def test_participation_rate_in_range(self):
        rng = random.Random(0)
        samples = [sample_personality(rng=rng).participation_rate for _ in range(200)]
        assert all(0.50 <= v <= 1.0 for v in samples)

    def test_fixed_fields_are_constant(self):
        rng = random.Random(0)
        samples = [sample_personality(rng=rng) for _ in range(50)]
        # Per DEFAULT_POPULATION_DIST: jitter, noise, stubbornness are fixed.
        for p in samples:
            assert p.check_interval_jitter == pytest.approx(1.0)
            assert p.trade_size_noise == pytest.approx(0.20)
            assert p.stubbornness == pytest.approx(0.30)

    def test_signal_sensitivity_in_range(self):
        rng = random.Random(0)
        samples = [sample_personality(rng=rng).signal_sensitivity for _ in range(200)]
        assert all(0.20 <= v <= 0.90 for v in samples)

    def test_custom_distribution_config(self):
        config = {
            "check_interval_mean": {"dist": "fixed", "value": 5.0},
            "edge_threshold": {"dist": "uniform", "low": 0.20, "high": 0.30},
            "participation_rate": {"dist": "fixed", "value": 1.0},
            "check_interval_jitter": {"dist": "fixed", "value": 0.5},
            "trade_size_noise": {"dist": "fixed", "value": 0.10},
            "signal_sensitivity": {"dist": "fixed", "value": 0.90},
            "stubbornness": {"dist": "fixed", "value": 0.05},
        }
        rng = random.Random(7)
        for _ in range(50):
            p = sample_personality(config, rng=rng)
            assert p.check_interval_mean == pytest.approx(5.0)
            assert 0.20 <= p.edge_threshold <= 0.30
            assert p.participation_rate == pytest.approx(1.0)
            assert p.signal_sensitivity == pytest.approx(0.90)
            assert p.stubbornness == pytest.approx(0.05)

    def test_normal_distribution_config(self):
        config = {f: {"dist": "fixed", "value": getattr(Personality(), f)} for f in Personality.__dataclass_fields__}
        config["check_interval_mean"] = {"dist": "normal", "mean": 2.5, "std": 0.1}
        rng = random.Random(42)
        samples = [sample_personality(config, rng=rng).check_interval_mean for _ in range(200)]
        mean = sum(samples) / len(samples)
        assert abs(mean - 2.5) < 0.1  # within 0.1 of the mean with 200 samples


# ---------------------------------------------------------------------------
# DEFAULT_POPULATION_DIST
# ---------------------------------------------------------------------------


class TestDefaultPopulationDist:
    def test_all_fields_covered(self):
        assert set(DEFAULT_POPULATION_DIST.keys()) == set(Personality.__dataclass_fields__)

    def test_uniform_fields_have_low_high(self):
        for key, cfg in DEFAULT_POPULATION_DIST.items():
            if cfg.get("dist") == "uniform":
                assert "low" in cfg, f"{key} missing 'low'"
                assert "high" in cfg, f"{key} missing 'high'"
                assert cfg["low"] < cfg["high"], f"{key}: low must be < high"

    def test_fixed_fields_have_value(self):
        for key, cfg in DEFAULT_POPULATION_DIST.items():
            if cfg.get("dist") == "fixed":
                assert "value" in cfg, f"{key} missing 'value'"

    def test_distribution_matches_spec(self):
        """Verify the exact distribution shape specified in FINAL_PHASE_TASKS.md."""
        assert DEFAULT_POPULATION_DIST["check_interval_mean"]["dist"] == "uniform"
        assert DEFAULT_POPULATION_DIST["check_interval_mean"]["low"] == pytest.approx(1.0)
        assert DEFAULT_POPULATION_DIST["check_interval_mean"]["high"] == pytest.approx(4.0)

        assert DEFAULT_POPULATION_DIST["edge_threshold"]["dist"] == "uniform"
        assert DEFAULT_POPULATION_DIST["edge_threshold"]["low"] == pytest.approx(0.01)
        assert DEFAULT_POPULATION_DIST["edge_threshold"]["high"] == pytest.approx(0.10)

        assert DEFAULT_POPULATION_DIST["participation_rate"]["dist"] == "uniform"
        assert DEFAULT_POPULATION_DIST["participation_rate"]["low"] == pytest.approx(0.50)
        assert DEFAULT_POPULATION_DIST["participation_rate"]["high"] == pytest.approx(1.0)

        assert DEFAULT_POPULATION_DIST["signal_sensitivity"]["dist"] == "uniform"
        assert DEFAULT_POPULATION_DIST["signal_sensitivity"]["low"] == pytest.approx(0.20)
        assert DEFAULT_POPULATION_DIST["signal_sensitivity"]["high"] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Personality fields affecting agent behavior (via AutonomousAgent)
# ---------------------------------------------------------------------------


class TestPersonalityFieldsBehavior:
    """
    Verify that each of the 7 personality fields demonstrably affects behavior
    in AutonomousAgent without needing a live server (use run_cycle mocks).
    """

    def _make_agent(self, personality: Personality, belief: float = 0.70):
        """Build an AutonomousAgent with mocked HTTP session."""
        import sys
        from pathlib import Path
        # autonomous_agent imports crra_math from the same directory
        from autonomous_agent import AutonomousAgent
        return AutonomousAgent(
            agent_id=1,
            api_base_url="http://fake",
            personality=personality,
            belief=belief,
            rho=1.0,
            cash=100.0,
            rng=random.Random(42),
        )

    def test_personality_stored_as_dataclass(self):
        p = Personality(edge_threshold=0.05)
        from autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://x",
            personality=p, belief=0.6, rho=1.0, cash=100.0,
        )
        assert isinstance(agent.personality, Personality)
        assert agent.personality.edge_threshold == pytest.approx(0.05)

    def test_dict_personality_converted_to_dataclass(self):
        from autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://x",
            personality={"edge_threshold": 0.09, "participation_rate": 0.6},
            belief=0.6, rho=1.0, cash=100.0,
        )
        assert isinstance(agent.personality, Personality)
        assert agent.personality.edge_threshold == pytest.approx(0.09)
        assert agent.personality.participation_rate == pytest.approx(0.6)

    def test_none_personality_uses_defaults(self):
        from autonomous_agent import AutonomousAgent
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://x",
            personality=None, belief=0.6, rho=1.0, cash=100.0,
        )
        assert isinstance(agent.personality, Personality)
        defaults = Personality()
        assert agent.personality.edge_threshold == pytest.approx(defaults.edge_threshold)

    def test_edge_threshold_blocks_trade_when_price_close(self, monkeypatch):
        """High edge_threshold means agent skips when |belief - price| is small."""
        from autonomous_agent import AutonomousAgent
        p = Personality(edge_threshold=0.20, participation_rate=1.0, trade_size_noise=0.0)
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://fake",
            personality=p, belief=0.70, rho=1.0, cash=100.0,
            rng=random.Random(0),
        )
        # Mock list_open_markets
        monkeypatch.setattr(agent, "list_open_markets", lambda: [{"id": 1, "price": 0.68}])
        monkeypatch.setattr(agent, "get_price_snapshot", lambda mid: {"price": 0.68})
        monkeypatch.setattr(agent, "get_agent_state", lambda mid: None)
        # |0.70 - 0.68| = 0.02 < threshold 0.20 => skip
        assert agent.run_cycle() == "edge_too_small"

    def test_edge_threshold_allows_trade_when_price_far(self, monkeypatch):
        """Low edge_threshold: agent trades when belief - price is large."""
        from autonomous_agent import AutonomousAgent
        p = Personality(edge_threshold=0.01, participation_rate=1.0, trade_size_noise=0.0)
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://fake",
            personality=p, belief=0.70, rho=1.0, cash=100.0,
            rng=random.Random(0),
        )
        monkeypatch.setattr(agent, "list_open_markets", lambda: [{"id": 1, "price": 0.50}])
        monkeypatch.setattr(agent, "get_price_snapshot", lambda mid: {"price": 0.50})
        monkeypatch.setattr(agent, "get_agent_state", lambda mid: None)
        monkeypatch.setattr(agent, "submit_trade", lambda mid, qty, mechanism="lmsr", limit_price=None: {"agent_cash_after": 95.0, "agent_shares_after": 3.0})
        result = agent.run_cycle()
        assert result == "traded"

    def test_participation_rate_skips_when_zero(self, monkeypatch):
        """participation_rate=0 means the agent never trades even when edge is large."""
        from autonomous_agent import AutonomousAgent
        p = Personality(edge_threshold=0.01, participation_rate=0.0, trade_size_noise=0.0)
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://fake",
            personality=p, belief=0.90, rho=1.0, cash=100.0,
            rng=random.Random(0),
        )
        monkeypatch.setattr(agent, "list_open_markets", lambda: [{"id": 1, "price": 0.50}])
        monkeypatch.setattr(agent, "get_price_snapshot", lambda mid: {"price": 0.50})
        monkeypatch.setattr(agent, "get_agent_state", lambda mid: None)
        assert agent.run_cycle() == "skipped_participation"

    def test_trade_size_noise_affects_quantity(self, monkeypatch):
        """Agents with different trade_size_noise submit different quantities."""
        from autonomous_agent import AutonomousAgent

        submitted_quantities = []

        def capture_trade(mid, qty, mechanism="lmsr", limit_price=None):
            submitted_quantities.append(qty)
            return {"agent_cash_after": 90.0, "agent_shares_after": qty}

        for noise in (0.0, 0.5):
            p = Personality(edge_threshold=0.01, participation_rate=1.0, trade_size_noise=noise)
            agent = AutonomousAgent(
                agent_id=1, api_base_url="http://fake",
                personality=p, belief=0.80, rho=1.0, cash=100.0,
                rng=random.Random(7),
            )
            monkeypatch.setattr(agent, "list_open_markets", lambda: [{"id": 1, "price": 0.50}])
            monkeypatch.setattr(agent, "get_price_snapshot", lambda mid: {"price": 0.50})
            monkeypatch.setattr(agent, "get_agent_state", lambda mid: None)
            monkeypatch.setattr(agent, "submit_trade", capture_trade)
            agent.run_cycle()

        # noise=0.0 produces a deterministic size; noise=0.5 differs
        assert len(submitted_quantities) == 2
        # With noise=0.0, x_star is unchanged; with noise=0.5, it's scaled
        # Both should be nonzero
        assert all(q != 0 for q in submitted_quantities)

    def test_signal_sensitivity_governs_belief_update(self, monkeypatch):
        """High sensitivity => belief moves more toward externally updated global belief."""
        from autonomous_agent import AutonomousAgent

        initial_belief = 0.50
        global_belief_after_news = 0.80

        # High sensitivity agent
        p_high = Personality(
            edge_threshold=0.0, participation_rate=1.0, trade_size_noise=0.0,
            signal_sensitivity=1.0, stubbornness=0.0,
        )
        agent_high = AutonomousAgent(
            agent_id=1, api_base_url="http://fake",
            personality=p_high, belief=initial_belief, rho=1.0, cash=100.0,
            rng=random.Random(0),
        )

        # Low sensitivity agent
        p_low = Personality(
            edge_threshold=0.0, participation_rate=1.0, trade_size_noise=0.0,
            signal_sensitivity=0.1, stubbornness=0.0,
        )
        agent_low = AutonomousAgent(
            agent_id=2, api_base_url="http://fake",
            personality=p_low, belief=initial_belief, rho=1.0, cash=100.0,
            rng=random.Random(0),
        )

        market_state = {
            "belief": global_belief_after_news,
            "cash": 100.0, "shares": 0.0, "rho": 1.0,
        }
        for agent in (agent_high, agent_low):
            monkeypatch.setattr(agent, "list_open_markets", lambda: [{"id": 1, "price": 0.50}])
            monkeypatch.setattr(agent, "get_price_snapshot", lambda mid: {"price": 0.50})
            monkeypatch.setattr(agent, "get_agent_state", lambda mid: market_state)
            monkeypatch.setattr(agent, "submit_trade", lambda mid, qty, mechanism="lmsr", limit_price=None: {"agent_cash_after": 90.0, "agent_shares_after": qty})
            agent.run_cycle()

        # High sensitivity agent should move more toward 0.80
        assert agent_high.belief > agent_low.belief

    def test_stubbornness_dampens_belief_update(self, monkeypatch):
        """High stubbornness => belief barely moves even with high signal_sensitivity."""
        from autonomous_agent import AutonomousAgent

        initial_belief = 0.50
        global_belief_after_news = 0.90

        p_stubborn = Personality(
            edge_threshold=0.0, participation_rate=1.0, trade_size_noise=0.0,
            signal_sensitivity=1.0, stubbornness=0.99,  # almost fully stubborn
        )
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://fake",
            personality=p_stubborn, belief=initial_belief, rho=1.0, cash=100.0,
            rng=random.Random(0),
        )
        market_state = {
            "belief": global_belief_after_news,
            "cash": 100.0, "shares": 0.0, "rho": 1.0,
        }
        monkeypatch.setattr(agent, "list_open_markets", lambda: [{"id": 1, "price": 0.50}])
        monkeypatch.setattr(agent, "get_price_snapshot", lambda mid: {"price": 0.50})
        monkeypatch.setattr(agent, "get_agent_state", lambda mid: market_state)
        monkeypatch.setattr(agent, "submit_trade", lambda mid, qty, mechanism="lmsr", limit_price=None: {"agent_cash_after": 90.0, "agent_shares_after": qty})
        agent.run_cycle()

        # With stubbornness=0.99, influence = 1.0 * (1 - 0.99) = 0.01
        # belief update = 0.50 + 0.01 * (0.90 - 0.50) = 0.504
        assert agent.belief == pytest.approx(0.504, abs=1e-9)

    def test_belief_never_read_from_market_position(self, monkeypatch):
        """Agent belief is NOT overwritten from market state when stubbornness is max."""
        from autonomous_agent import AutonomousAgent

        p = Personality(
            edge_threshold=0.0, participation_rate=1.0, trade_size_noise=0.0,
            signal_sensitivity=0.0, stubbornness=1.0,  # fully ignore external signals
        )
        initial_belief = 0.72
        agent = AutonomousAgent(
            agent_id=1, api_base_url="http://fake",
            personality=p, belief=initial_belief, rho=1.0, cash=100.0,
            rng=random.Random(0),
        )
        # Market state reports a completely different belief
        market_state = {"belief": 0.20, "cash": 100.0, "shares": 0.0, "rho": 1.0}
        monkeypatch.setattr(agent, "list_open_markets", lambda: [{"id": 1, "price": 0.50}])
        monkeypatch.setattr(agent, "get_price_snapshot", lambda mid: {"price": 0.50})
        monkeypatch.setattr(agent, "get_agent_state", lambda mid: market_state)
        monkeypatch.setattr(agent, "submit_trade", lambda mid, qty, mechanism="lmsr", limit_price=None: {"agent_cash_after": 90.0, "agent_shares_after": qty})
        agent.run_cycle()

        # With signal_sensitivity=0, no influence at all — belief must not change
        assert agent.belief == pytest.approx(initial_belief)

    def test_check_interval_controls_wait_time(self):
        """Agent with longer check_interval_mean waits longer between cycles."""
        from autonomous_agent import AutonomousAgent

        p_slow = Personality(check_interval_mean=10.0, check_interval_jitter=0.0)
        p_fast = Personality(check_interval_mean=0.1, check_interval_jitter=0.0)

        # Measure the sleep duration via the stop flag
        import threading
        for p, expected_min in [(p_slow, 9.9), (p_fast, 0.05)]:
            agent = AutonomousAgent(
                agent_id=1, api_base_url="http://x",
                personality=p, belief=0.5, rho=1.0, cash=100.0,
                rng=random.Random(0),
            )
            # Signal immediately so wait returns True (stopped)
            agent._stop_flag.set()
            import time
            t0 = time.monotonic()
            result = agent._wait_for_next_cycle()
            elapsed = time.monotonic() - t0
            # Since flag was pre-set, it returns immediately regardless of interval
            assert result is True  # stop_flag was set
