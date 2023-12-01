import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

# Load the CSV file
file_path = 'backtest/data/USD_JPY.csv'
forex_data = pd.read_csv(file_path)

# Selecting the 'Close' column for analysis
close_prices = forex_data['Close'].values

# Choose a wavelet function and the level of decomposition
wavelet_name = 'db1'
level = 4

# Decompose the signal using Discrete Wavelet Transform
coeffs = pywt.wavedec(close_prices, wavelet_name, level=level)
cA = coeffs[0]  # Approximation coefficients
cD = coeffs[1:]  # Detail coefficients

print("coeffs: ", coeffs)

# Plotting the original signal
plt.figure(figsize=(12, 6))
plt.plot(close_prices, label='Original Signal')
plt.title('Original USD/JPY Close Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Plotting the approximation and details
plt.figure(figsize=(12, 6))
plt.subplot(len(coeffs), 1, 1)
plt.plot(cA, label='Approximation')
plt.title('Approximation Coefficients')
plt.legend()
plt.grid()

for i, coeff in enumerate(cD, start=1):
    plt.subplot(len(coeffs), 1, i+1)
    plt.plot(coeff, label=f'Detail {i}')
    plt.title(f'Detail Coefficients {i}')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

# Reconstruct the signal using the significant coefficients
reconstructed_signal = pywt.waverec(coeffs, wavelet_name)

# Plotting the reconstructed signal
plt.figure(figsize=(12, 6))
plt.plot(reconstructed_signal, label='Reconstructed Signal')
plt.plot(close_prices, label='Original Signal', alpha=0.5)
plt.title('Reconstructed vs Original Signal')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
