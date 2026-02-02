from market_logic import LMSRMarketMaker

def test():
    # initialize with b=100
    m = LMSRMarketMaker(b=100)
    
    # check initial price 
    p_initial = m.get_price()
    
    # cxecute trade
    cost = m.calculate_trade_cost(10)
    p_final = m.get_price()
    
    # 4. check
    if p_final > p_initial:
        print("\n Success")
    else:
        print("\n Error.")

if __name__ == "__main__":
    test()