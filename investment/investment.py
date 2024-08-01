# Example investment.py content

# List of example mutual funds and government schemes
mutual_funds = [
    {"name": "Equity Fund A", "annual_return": 0.12},
    {"name": "Debt Fund B", "annual_return": 0.08},
    {"name": "Liquid Fund C", "annual_return": 0.04}  # Example liquid fund for emergency funds
]

government_schemes = [
    {"name": "PPF", "annual_return": 0.075, "lock_in_period": 15},
    {"name": "Sukanya Samriddhi Yojana", "annual_return": 0.077, "lock_in_period": 21}
]

def calculate_returns(principal, annual_return, years):
    # Compound interest formula
    return principal * ((1 + annual_return) ** years)

def suggest_investments(amount, years, risk_tolerance, financial_goal, age, current_savings, emergency_fund_ratio=0.1):
    # Determine split based on risk tolerance and financial goal
    if risk_tolerance == 'high' and financial_goal == 'growth':
        mf_ratio = 0.7
        gs_ratio = 0.3
    elif risk_tolerance == 'medium' or financial_goal == 'income':
        mf_ratio = 0.5
        gs_ratio = 0.5
    else:  # low risk tolerance or stability goal
        mf_ratio = 0.3
        gs_ratio = 0.7

    # Adjust allocations based on age and current financial situation
    if age < 35 and current_savings >= amount * 2:
        mf_ratio += 0.1
        gs_ratio -= 0.1
    elif age > 50 or current_savings < amount:
        mf_ratio -= 0.1
        gs_ratio += 0.1

    # Allocate a portion for emergency funds in liquid mutual funds
    emergency_fund_amount = amount * emergency_fund_ratio
    investable_amount = amount - emergency_fund_amount

    mf_amount = investable_amount * mf_ratio
    gs_amount = investable_amount * gs_ratio

    # Selecting the best mutual fund and government scheme based on annual return
    best_mf = max(mutual_funds, key=lambda x: x["annual_return"] if x["name"] != "Liquid Fund C" else 0)
    best_gs = max(government_schemes, key=lambda x: x["annual_return"])

    # Selecting a liquid fund for emergency funds
    liquid_fund = next(fund for fund in mutual_funds if fund["name"] == "Liquid Fund C")

    # Calculating the returns
    mf_returns = calculate_returns(mf_amount, best_mf["annual_return"], years)
    gs_returns = calculate_returns(gs_amount, best_gs["annual_return"], years)
    liquid_fund_returns = calculate_returns(emergency_fund_amount, liquid_fund["annual_return"], years)

    total_returns = mf_returns + gs_returns + liquid_fund_returns

    return best_mf, best_gs, liquid_fund, total_returns, emergency_fund_amount, mf_amount, gs_amount
