from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Load the Forex data
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, '../backtest/data/USD_JPY_YF.csv')
forex_data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
forex_data['Date'] = pd.to_datetime(forex_data['Date'])

# Selecting the 'Close' column for analysis
close_prices = forex_data['Close'].to_numpy()

# Finding peaks
peaks, _ = find_peaks(close_prices)

# Finding troughs by inverting the data
troughs, _ = find_peaks(-1 * close_prices)

# Plotting the results with reduced marker size
plt.figure(figsize=(12, 6))
plt.plot(forex_data['Date'], close_prices, label='Close Prices')
plt.plot(forex_data['Date'].iloc[peaks], close_prices[peaks], 'r*', markersize=1, label='Peaks')  # Reduce marker size by 50%
plt.plot(forex_data['Date'].iloc[troughs], close_prices[troughs], 'g*', markersize=1, label='Troughs')  # Reduce marker size by 50%
plt.title('Peaks and Troughs in USD/JPY Close Prices')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()
plt.grid()
plt.xticks(pd.date_range(start=forex_data['Date'].iloc[0], \
                         end=forex_data['Date'].iloc[-1], freq='Y'), \
                         fontsize=8)  # Set ticks to yearly with reduced font size

# Reduce the number of gridlines on the x-axis
plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Set major ticks as years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as years
# Adjust the number of minor ticks if needed
plt.gca().xaxis.set_minor_locator(mdates.YearLocator(month=6))  # Set minor ticks every 6 months

plt.show()
