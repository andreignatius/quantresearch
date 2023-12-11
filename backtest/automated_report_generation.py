import pandas as pd
import warnings
from logreg_oop import rolling_window_train_predict as rolling_window_train_predict_logreg
from gbt_oop import rolling_window_train_predict as rolling_window_train_predict_gbt
import os
import matplotlib.pyplot as plt
import json

# Suppress warnings
warnings.filterwarnings("ignore")

# Generate leverage_factor values from 1 to 3 in increments of 1
# leverage_factors = [x for x in range(1, 4)]

# Generate leverage_factor values from 1 to 5 in increments of 0.5
# leverage_factors = [x / 2 for x in range(2, 11)]

# Generate leverage factors
leverage_factors = [1, 1.5, 2, 3, 4]
annual_interest_rate = 0.05

logreg_results = {}

# Load the Forex data
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data/USD_JPY_YF.csv')
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
data.sort_values('Date', inplace=True)       # Sort by 'Date'

# Create an empty DataFrame to store leverage and P&L
pnl_data = pd.DataFrame(columns=['Leverage', 'PnL', 'Interest Costs', 'Transaction Costs', '% Returns per Period', '% Returns per Annum'])

# Cycle through leverage factors
for leverage_factor in leverage_factors:
    print("LEVERAGE FACTOR LOGREG: " , leverage_factor)
    # Apply the rolling window approach to obtain PnL values
    trade_logs, final_values, interest_costs_total, transaction_costs_total = rolling_window_train_predict_logreg(data, 2013, 2023, 12, 6, leverage_factor=leverage_factor, annual_interest_rate=annual_interest_rate)  # 12 months training, 6 months testing

    pnl_per_quarter = [x - 10000 for x in final_values]
    total_pnl = sum(pnl_per_quarter)

    print("final trade_logs: ", trade_logs)
    print("final_values: ", final_values)
    print("pnl_per_quarter: ", pnl_per_quarter)
    print("final_pnl: ", total_pnl )
    print("interest_costs: ", interest_costs_total)
    print("transaction_costs :", transaction_costs_total)
    percentage_returns = ( sum(pnl_per_quarter) / len(pnl_per_quarter) ) / 10000 * 100
    print("percentage_returns per period: ", percentage_returns)
    print("percentage_returns per annum: ", percentage_returns * 2)

    # Append leverage and P&L to the DataFrame using pd.concat
    new_row = pd.DataFrame({ 
    	'Leverage': [leverage_factor], 
    	'PnL': [total_pnl],
    	'Interest Costs': [sum(interest_costs_total)],
    	'Transaction Costs': [sum(transaction_costs_total)],
    	'% Returns per Period': [percentage_returns], 
    	'% Returns per Annum': [percentage_returns * 2],
    	})
    logreg_results_factor = {
    	'PnL': total_pnl,
    	'PnL Distribution' : pnl_per_quarter,
    	'Interest Costs': sum(interest_costs_total),
    	'Transaction Costs': sum(transaction_costs_total),
    	'% Returns per Period': percentage_returns, 
    	'% Returns per Annum': percentage_returns * 2,
    	}

    logreg_results['Leverage' + str(leverage_factor)] = logreg_results_factor

    pnl_data = pd.concat([pnl_data, new_row], ignore_index=True)

    # Display the DataFrame
    print(pnl_data)
    # Save DataFrame to a CSV file
    pnl_data.to_csv('logreg_pnl_v_leverage.csv', index=False)
    # save Data results into JSON file
    with open('logreg_pnl_v_leverage.json', 'w', encoding='utf-8') as f:
    	json.dump(logreg_results, f, ensure_ascii=False, indent=4)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(pnl_data['Leverage'], pnl_data['PnL'], marker='o', linestyle='-')
plt.title('PnL vs Leverage')
plt.xlabel('Leverage')
plt.ylabel('PnL')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('logreg_pnl_v_leverage.png')
plt.show()


gbt_results = {}
# Create an empty DataFrame to store leverage and P&L
pnl_data = pd.DataFrame(columns=['Leverage', 'PnL', 'Interest Costs', 'Transaction Costs', '% Returns per Period', '% Returns per Annum'])

# Cycle through leverage factors
for leverage_factor in leverage_factors:
    print("LEVERAGE FACTOR GBT: " , leverage_factor)
    # Apply the rolling window approach to obtain PnL values
    trade_logs, final_values, interest_costs_total, transaction_costs_total = rolling_window_train_predict_gbt(data, 2013, 2023, 12, 6, leverage_factor=leverage_factor, annual_interest_rate=annual_interest_rate)  # 12 months training, 6 months testing

    pnl_per_quarter = [x - 10000 for x in final_values]
    total_pnl = sum(pnl_per_quarter)

    print("final trade_logs: ", trade_logs)
    print("final_values: ", final_values)
    print("pnl_per_quarter: ", pnl_per_quarter)
    print("final_pnl: ", total_pnl )
    print("interest_costs: ", interest_costs_total)
    print("transaction_costs :", transaction_costs_total)
    percentage_returns = ( sum(pnl_per_quarter) / len(pnl_per_quarter) ) / 10000 * 100
    print("percentage_returns per period: ", percentage_returns)
    print("percentage_returns per annum: ", percentage_returns * 2)

    # Append leverage and P&L to the DataFrame using pd.concat
    new_row = pd.DataFrame({ 
    	'Leverage': [leverage_factor], 
    	'PnL': [total_pnl],
    	'Interest Costs': [sum(interest_costs_total)],
    	'Transaction Costs': [sum(transaction_costs_total)],
    	'% Returns per Period': [percentage_returns], 
    	'% Returns per Annum': [percentage_returns * 2],
    	})
    gbt_results_factor = {
    	'PnL': total_pnl,
    	'PnL Distribution' : pnl_per_quarter,
    	'Interest Costs': sum(interest_costs_total),
    	'Transaction Costs': sum(transaction_costs_total),
    	'% Returns per Period': percentage_returns, 
    	'% Returns per Annum': percentage_returns * 2,
    	}

    gbt_results['Leverage' + str(leverage_factor)] = gbt_results_factor
    pnl_data = pd.concat([pnl_data, new_row], ignore_index=True)

    # Display the DataFrame
    print(pnl_data)
    # Save DataFrame to a CSV file
    pnl_data.to_csv('gbt_pnl_v_leverage.csv', index=False)
    # save Data results into JSON file
    with open('gbt_pnl_v_leverage.json', 'w', encoding='utf-8') as f:
    	json.dump(gbt_results, f, ensure_ascii=False, indent=4)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(pnl_data['Leverage'], pnl_data['PnL'], marker='o', linestyle='-')
plt.title('PnL vs Leverage')
plt.xlabel('Leverage')
plt.ylabel('PnL')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.savefig('gbt_pnl_v_leverage.png')
plt.show()

#### ONE LAST PART


















