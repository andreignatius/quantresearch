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