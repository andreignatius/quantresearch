from scipy.signal import find_peaks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Load the Forex data
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, '../backtest/final_dataset_with_new_features.csv')
forex_data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
forex_data['Date'] = pd.to_datetime(forex_data['Date'])

# Plotting the results with reduced marker size
plt.figure(figsize=(12, 6))
plt.plot(forex_data['Date'], forex_data['HurstExponent'], label='Hurst Exponent Estimates')
plt.axhline(y=0.5, color='r', linestyle='--', label='Trend Reference Line')
plt.title('Hurst Exponent Estimates')
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
