from flask import Flask, render_template, request
from investment import suggest_investments

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        amount = float(request.form['amount'])
        years = int(request.form['years'])
        risk_tolerance = request.form['risk_tolerance'].lower()
        financial_goal = request.form['financial_goal'].lower()
        age = int(request.form['age'])
        current_savings = float(request.form['current_savings'])
        
        best_mf, best_gs, liquid_fund, total_returns, emergency_fund_amount, mf_amount, gs_amount = suggest_investments(amount, years, risk_tolerance, financial_goal, age, current_savings)
        
        return render_template('result.html', 
                               amount=amount, 
                               years=years, 
                               risk_tolerance=risk_tolerance, 
                               financial_goal=financial_goal, 
                               age=age, 
                               current_savings=current_savings,
                               best_mf=best_mf,
                               mf_amount=mf_amount,
                               best_gs=best_gs,
                               gs_amount=gs_amount,
                               liquid_fund=liquid_fund,
                               emergency_fund_amount=emergency_fund_amount,
                               total_returns=total_returns)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
