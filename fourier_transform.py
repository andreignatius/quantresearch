import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import find_peaks

# Load the CSV file
file_path = 'USD_JPY.csv'  # Replace with your file path
forex_data = pd.read_csv(file_path)

# Selecting the 'Close' column for analysis
close_prices = forex_data['Close'].to_numpy()

# Number of data points and period (assuming daily data)
N = len(close_prices)
T = 1.0  # 1 day

# Compute the Fast Fourier Transform (FFT)
close_fft = fft(close_prices)
fft_freq = np.fft.fftfreq(N, T)

# Extracting positive frequencies and corresponding FFT values
positive_frequencies = fft_freq[:N//2]
positive_fft_values = 2.0/N * np.abs(close_fft[0:N//2])

# # Identifying the peaks (dominant frequencies)
# peaks, _ = find_peaks(positive_fft_values, height=0)
# dominant_frequencies = positive_frequencies[peaks]
# dominant_amplitudes = positive_fft_values[peaks]

# # Plotting the FFT results with identified peaks
# plt.figure(figsize=(12, 6))
# plt.plot(positive_frequencies, positive_fft_values)
# plt.plot(dominant_frequencies, dominant_amplitudes, 'r*')
# plt.title('Dominant Frequencies in USD/JPY Close Prices')
# plt.xlabel('Frequency (cycles per day)')
# plt.ylabel('Amplitude')
# plt.grid()
# plt.show()

# Filtering out frequencies lower than 0.02
min_frequency_threshold = 0.02
frequency_range = (0, 0.1)  # Adjust as needed
amplitude_threshold = 0.1  # Adjust as needed

filtered_frequencies = positive_frequencies[(positive_frequencies >= min_frequency_threshold) & (positive_frequencies <= frequency_range[1])]
filtered_amplitudes = positive_fft_values[(positive_frequencies >= min_frequency_threshold) & (positive_frequencies <= frequency_range[1])]

# Identifying significant frequencies above the amplitude threshold
significant_peaks, _ = find_peaks(filtered_amplitudes, height=amplitude_threshold)
significant_frequencies = filtered_frequencies[significant_peaks]
significant_amplitudes = filtered_amplitudes[significant_peaks]

# Plotting the refined FFT results with discounted frequencies
plt.figure(figsize=(12, 6))
plt.plot(filtered_frequencies, filtered_amplitudes, label='Filtered FFT Amplitudes')
plt.plot(significant_frequencies, significant_amplitudes, 'r*', label='Significant Frequencies')
plt.title('Significant Frequencies in USD/JPY Close Prices (Filtered, >0.02)')
plt.xlabel('Frequency (cycles per day)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# significant_frequencies, significant_amplitudes

# Converting frequencies to days per cycle
days_per_cycle = 1 / filtered_frequencies
significant_days_per_cycle = 1 / significant_frequencies

# Plotting the refined FFT results with days per cycle
plt.figure(figsize=(12, 6))
plt.plot(days_per_cycle, filtered_amplitudes, label='Filtered FFT Amplitudes')
plt.plot(significant_days_per_cycle, significant_amplitudes, 'r*', label='Significant Cycles')
plt.title('Significant Cycles in USD/JPY Close Prices (Filtered, >0.02 cycles/day)')
plt.xlabel('Cycle Length (days per cycle)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

# significant_days_per_cycle, significant_amplitudes

