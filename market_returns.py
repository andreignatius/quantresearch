# Import Libraries
import pandas as pd
import subprocess
import re

# Load the Forex data
file_path = 'backtest/data/USD_JPY.csv'
forex_data = pd.read_csv(file_path)

# Generate Market Return
starting_price = forex_data['Close'].iloc[0]
final_price = forex_data['Close'].iloc[-1]
investment = 10000
gains = (final_price - starting_price)/starting_price
rm = gains * investment

# Line Break
print()

# Generate Output
print("Starting Price: $", round(starting_price,2))
print("Final Price: $", round(final_price,2))
print("Final Market Return: ", round(gains*100,2),"%")
print("Final Portfolio Value: $", round((1+gains)*investment,2) )

# Line Break
print()

# Calculate Excess Returns for Performance Benchmark
output = subprocess.check_output(['python', 'performance_benchmark.py'], universal_newlines=True)
match = re.search(r"Final Portfolio Value:\s*([\d.]+)", output)
if match:
    final_portfolio_value = float(match.group(1))
    pb_excess = final_portfolio_value - (1+gains)*investment
    print("Performance Benchmark Returns: $", round(final_portfolio_value,2))
    print("Performance Benchmark Excess Returns: $", round(pb_excess,2))
else:
    pass

# Line Break
print()

# Calculate Excess Returns for Logistic Regression
output = subprocess.check_output(['python', 'backtest\logreg_oop.py'], universal_newlines=True)
match = re.search(r"Final Portfolio Value:\s*([\d.]+)", output)
if match:
    final_portfolio_value = float(match.group(1))
    pb_excess = final_portfolio_value - (1+gains)*investment
    print("Logistic Regression Returns: $", round(final_portfolio_value,2))
    print("Logistic Regression Excess Returns: $", round(pb_excess,2))
else:
    pass




