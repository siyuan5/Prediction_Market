import numpy as np

class LMSRMarketMaker:
    # handles the LSMR math for the "house"

    def __init__(self, b, initial_inventory=[0, 0]):
        self.b = b
        self.inventory = np.array(initial_inventory, dtype=float)

    def get_cost(self, inventory):
        # calculate total cost 
        # formula: C = b * ln(exp(q1/b) + exp(q0/b))
        sum_exp = np.sum(np.exp(inventory / self.b))
        return self.b * np.log(sum_exp)

    def get_price(self):
        # returns the price of outcome 1 
        # formula: p1 = exp(q1/b) / (exp(q1/b) + exp(q0/b))
        exponents = np.exp(self.inventory / self.b)
        price_q1 = exponents[0] / np.sum(exponents)
        return price_q1

    def calculate_trade_cost(self, delta_q1):
        # update inventory and return cost the agent must pay
        # cost is difference in cost function before and after trade
        
        #determine new state
        old_cost = self.get_cost(self.inventory)
        new_inventory = self.inventory.copy()
        new_inventory[0] += delta_q1
        new_cost = self.get_cost(new_inventory)
        
        #final trade cost calculation
        trade_price = new_cost - old_cost
        self.inventory = new_inventory
        return trade_price