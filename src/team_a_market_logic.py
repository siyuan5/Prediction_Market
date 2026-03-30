"""
LMSR (logarithmic market scoring rule) automated market maker for binary outcomes.

The cost function C(q) = b * ln(exp(q1/b) + exp(q0/b)) makes the quoted price for
outcome 1 a proper probability p1 in (0, 1). Inventory `inventory[0]` is net YES
(q=1) exposure from the MM’s perspective; trades move inventory and charge agents
the *change* in C (no free lunch).
"""

import numpy as np


class LMSRMarketMaker:
    """Binary LMSR: maintains `inventory` [q1, q0] and exposes cost-based pricing."""

    def __init__(self, b, initial_inventory=[0, 0]):
        self.b = b
        self.inventory = np.array(initial_inventory, dtype=float)

    def get_cost(self, inventory):
        # Total liability to pay all outcomes: C = b * ln(exp(q1/b) + exp(q0/b))
        sum_exp = np.sum(np.exp(inventory / self.b))
        return self.b * np.log(sum_exp)

    def get_price(self):
        # Implied P(YES) = softmax of scaled inventories: p1 = e^{q1/b} / (e^{q1/b}+e^{q0/b})
        exponents = np.exp(self.inventory / self.b)
        price_q1 = exponents[0] / np.sum(exponents)
        return price_q1

    def calculate_trade_cost(self, delta_q1):
        # Agent buys delta_q1 YES shares → MM inventory[0] increases; they pay ΔC.
        old_cost = self.get_cost(self.inventory)
        new_inventory = self.inventory.copy()
        new_inventory[0] += delta_q1
        new_cost = self.get_cost(new_inventory)
        
        #final trade cost calculation
        trade_price = new_cost - old_cost
        self.inventory = new_inventory
        return trade_price