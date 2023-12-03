import yfinance as yf
import pandas as pd

# # Define the ticker symbol for USD to JPY exchange rate
# ticker_symbol = 'USDJPY=X'

# Define the ticker symbol for JPY to USD exchange rate
ticker_symbol = 'JPYUSD=X'

# Fetch data from Yahoo Finance
data = yf.download(ticker_symbol, start='1990-01-01', end='2023-12-01')

# Reset index to have Date as a column
data.reset_index(inplace=True)

# Save the data to a CSV file under 'backtest/data'
file_path = 'backtest/data/JPY_USD_YF.csv'
data.to_csv(file_path, index=False)

print(f"Exchange rate data saved to {file_path}")
