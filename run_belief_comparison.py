import sys
sys.path.insert(0, "src")

from simulation_engine import SimulationEngine
from belief_init import BeliefSpec

configs = [
    ("gaussian (baseline)", BeliefSpec(mode="gaussian", sigma=0.10)),
    ("bimodal (0.3 vs 0.9)", BeliefSpec(mode="bimodal", group_beliefs=[0.3, 0.9])),
    ("fixed at 0.4",         BeliefSpec(mode="fixed", fixed_value=0.4)),
]

print(f"{'config':<25} {'final price':>12} {'error vs P*':>12} {'mean init belief':>18}")
print("-" * 70)

for label, spec in configs:
    engine = SimulationEngine(
        mechanism="lmsr",
        phase=1,
        seed=42,
        ground_truth=0.70,
        n_agents=50,
        belief_spec=spec,
        b=500.0,
    )
    engine.run(200)
    state = engine.get_state()
    print(f"{label:<25} {state['price']:>12.4f} {state['error']:>12.4f} {engine.mean_initial_belief:>18.4f}")
