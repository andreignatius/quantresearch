from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks

# Load the CSV file
file_path = 'backtest/data/USD_JPY.csv'
forex_data = pd.read_csv(file_path)

# Selecting the 'Close' column for analysis
close_prices = forex_data['Close'].to_numpy()

# Finding peaks
peaks, _ = find_peaks(close_prices)

# Finding troughs by inverting the data
troughs, _ = find_peaks(-1 * close_prices)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(close_prices, label='Close Prices')
plt.plot(peaks, close_prices[peaks], 'r*', label='Peaks')
plt.plot(troughs, close_prices[troughs], 'g*', label='Troughs')
plt.title('Peaks and Troughs in USD/JPY Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.show()
