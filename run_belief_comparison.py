import sys
sys.path.insert(0, "src")  # Ensure src directory is on the Python path for module imports

from simulation_engine import SimulationEngine
from belief_init import BeliefSpec

# Define initial belief configurations to compare their impact in Phase 1 under LMSR
configs = [
    ("gaussian (baseline)", BeliefSpec(mode="gaussian", sigma=0.10)),
    ("bimodal (0.3 vs 0.9)", BeliefSpec(mode="bimodal", group_beliefs=[0.3, 0.9])),
    ("fixed at 0.4",         BeliefSpec(mode="fixed", fixed_value=0.4)),
]

print(f"{'config':<25} {'final price':>12} {'error vs P*':>12} {'mean init belief':>18}")
print("-" * 70)

for label, spec in configs:
    # Set up and run the simulation with the given prior/belief spec
    engine = SimulationEngine(
        mechanism="lmsr",
        phase=1,
        seed=42,
        ground_truth=0.70,
        n_agents=50,
        belief_spec=spec,
        b=500.0,  # LMSR liquidity parameter
    )
    engine.run(200)  # Run market for 200 rounds
    state = engine.get_state()
    # Print configuration label, final market price, error to true value, and mean initial agent belief
    print(f"{label:<25} {state['price']:>12.4f} {state['error']:>12.4f} {engine.mean_initial_belief:>18.4f}")
