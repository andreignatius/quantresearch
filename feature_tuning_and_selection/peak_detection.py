from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the CSV file
# Note: Replace '../backtest/data/USD_JPY_YF.csv' with your correct file path
file_path = '../backtest/data/USD_JPY_YF.csv'
forex_data = pd.read_csv(file_path)

# Assuming the 'Date' column exists in your CSV, replace 'Date' with the correct column name
dates = forex_data['Date']

# Selecting the 'Close' column for analysis
close_prices = forex_data['Close'].to_numpy()

# Finding peaks
peaks, _ = find_peaks(close_prices)

# Finding troughs by inverting the data
troughs, _ = find_peaks(-1 * close_prices)

# Create a DataFrame to store the Date and Peak/Trough information
peak_trough_data = pd.DataFrame({'Date': dates})
peak_trough_data['Peak'] = False
peak_trough_data['Trough'] = False

# Mark peaks and troughs in the DataFrame
peak_trough_data.loc[peaks, 'Peak'] = True
peak_trough_data.loc[troughs, 'Trough'] = True

# Save the DataFrame to a CSV file
peak_trough_data.to_csv('peak_detection.csv', index=False)
