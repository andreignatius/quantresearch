import pandas as pd
import numpy as np
from pykalman import KalmanFilter

# Load the data
file_path = 'backtest/data/USD_JPY.csv'
data = pd.read_csv(file_path)

# Select the 'Close' price for Kalman Filter application
close_prices = data['Close']

# Construct a Kalman Filter
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

# Use the observed data (close prices) to estimate the state
state_means, _ = kf.filter(close_prices.values)

# Convert state means to a Pandas Series for easy plotting
kalman_estimates = pd.Series(state_means.flatten(), index=data.index)

# Combine the original close prices and Kalman Filter estimates
combined_data = pd.DataFrame({
    'Close': close_prices,
    'Kalman Filter Estimate': kalman_estimates
})

print(combined_data.head())  # Display the first few rows of the dataframe
