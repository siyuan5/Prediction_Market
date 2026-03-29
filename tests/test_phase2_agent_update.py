import sys
from pathlib import Path

# Add src directory to import path for direct module imports in tests
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crra_agent import CRRAAgent
from team_b_crra_agent import TeamBCRRAAgent


def test_team_a_agent_update_belief_changes_value():
    # Team A: agent uses "weighted" update, belief should change & remain in [0.01, 0.99]
    a = CRRAAgent(agent_id=0, initial_cash=100.0, belief_p=0.2, rho=1.0)
    old = a.belief
    a.update_belief(0.9, method="weighted", w=0.5)
    assert a.belief != old  # Belief changes after update
    assert 0.01 <= a.belief <= 0.99  # Belief bounded away from 0, 1 (avoids degenerate values)


def test_team_b_agent_update_belief_changes_value():
    # Team B: "beta" update method with prior+obs strength should shift belief & remain in (0.01, 0.99)
    a = TeamBCRRAAgent(agent_id=0, initial_cash=100.0, belief_p=0.2, rho=1.0)
    old = a.belief
    a.update_belief(0.9, method="beta", prior_strength=20.0, obs_strength=10.0)
    assert a.belief != old  # Belief should update given new signal
    assert 0.01 <= a.belief <= 0.99  # Avoid numerically degenerate beliefs
