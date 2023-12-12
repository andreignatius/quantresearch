import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Download S&P 500 data from Yahoo Finance
sp500_data = yf.download('^GSPC', start='2013-01-01', end='2023-12-31')['Close']

# Limit data from 2013 to 2022
sp500_data = sp500_data.loc['2013-01-01':'2022-12-31']

# Calculate daily returns for S&P 500
sp500_returns = sp500_data.pct_change().dropna()

# Calculate average annualized returns for S&P 500
annualized_sp500_return = sp500_returns.mean() * 252  # 252 trading days in a year

# Assume risk-free rate (e.g., US Treasury yield for a similar period)
risk_free_rate = 0.02  # Replace with actual risk-free rate

# Calculate annualized standard deviation
sp500_std_dev = sp500_returns.std() * np.sqrt(252)  # Annualized standard deviation

# Calculate excess returns for S&P 500
sp500_excess_returns = sp500_returns - risk_free_rate

# Calculate Sharpe Ratio for S&P 500
sp500_sharpe_ratio = np.sqrt(252) * sp500_excess_returns.mean() / sp500_excess_returns.std()

######################################
#### PRINT RESULTS
######################################

print("S&P 500 Average Annualized Return (2013-2022):", annualized_sp500_return)
print("S&P 500 Standard Deviation of Returns (2013-2022):", sp500_std_dev)
print("S&P 500 Sharpe Ratio (2013-2022):", sp500_sharpe_ratio)

##############################################################################################################

# Download USD/JPY forex data from Yahoo Finance
usdjpy_data = yf.download('USDJPY=X', start='2013-01-01', end='2023-12-31')['Close']

# Limit data from 2013 to 2022
usdjpy_data = usdjpy_data.loc['2013-01-01':'2022-12-31']

# Calculate daily returns for USD/JPY
usdjpy_returns = usdjpy_data.pct_change().dropna()

# Calculate average annualized returns for USD/JPY
annualized_usdjpy_return = usdjpy_returns.mean() * 252  # 252 trading days in a year

# Assume risk-free rate (e.g., US Treasury yield for a similar period)
risk_free_rate = 0.02  # Replace with actual risk-free rate

# Calculate annualized standard deviation
usdjpy_std_dev = usdjpy_returns.std() * np.sqrt(252)  # Annualized standard deviation

# Calculate excess returns for USD/JPY
usdjpy_excess_returns = usdjpy_returns - risk_free_rate

# Calculate Sharpe Ratio for USD/JPY
usdjpy_sharpe_ratio = np.sqrt(252) * usdjpy_excess_returns.mean() / usdjpy_excess_returns.std()

######################################
#### PRINT RESULTS
######################################

print("\nUSD/JPY Average Annualized Return (2013-2022):", annualized_usdjpy_return)
print("USD/JPY Standard Deviation of Returns (2013-2022):", usdjpy_std_dev)
print("USD/JPY Sharpe Ratio (2013-2022):", usdjpy_sharpe_ratio)
