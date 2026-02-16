from crra_agent import CRRAAgent
from team_b_crra_agent import TeamBCRRAAgent


def test_team_a_agent_update_belief_changes_value():
    a = CRRAAgent(agent_id=0, initial_cash=100.0, belief_p=0.2, rho=1.0)
    old = a.belief
    a.update_belief(0.9, method="weighted", w=0.5)
    assert a.belief != old
    assert 0.01 <= a.belief <= 0.99


def test_team_b_agent_update_belief_changes_value():
    a = TeamBCRRAAgent(agent_id=0, initial_cash=100.0, belief_p=0.2, rho=1.0)
    old = a.belief
    a.update_belief(0.9, method="beta", prior_strength=20.0, obs_strength=10.0)
    assert a.belief != old
    assert 0.01 <= a.belief <= 0.99
