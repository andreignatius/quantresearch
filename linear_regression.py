import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the Forex data
file_path = 'USD_JPY.csv'
forex_data = pd.read_csv(file_path)

# Fourier Transform Analysis
close_prices = forex_data['Close'].to_numpy()
N = len(close_prices)
T = 1.0  # 1 day
close_fft = fft(close_prices)
fft_freq = np.fft.fftfreq(N, T)
positive_frequencies = fft_freq[:N//2]
positive_fft_values = 2.0/N * np.abs(close_fft[0:N//2])
amplitude_threshold = 0.1  # This can be adjusted
significant_peaks, _ = find_peaks(positive_fft_values, height=amplitude_threshold)
significant_frequencies = positive_frequencies[significant_peaks]
significant_amplitudes = positive_fft_values[significant_peaks]
days_per_cycle = 1 / significant_frequencies
fft_features = pd.DataFrame({
    'Frequency': significant_frequencies,
    'Amplitude': significant_amplitudes,
    'DaysPerCycle': days_per_cycle
})

# Peak and Trough Detection for Labeling
peaks, _ = find_peaks(close_prices)
troughs, _ = find_peaks(-1 * close_prices)
labels = pd.DataFrame(index=range(len(close_prices)), columns=['Label'])
labels.loc[peaks, 'Label'] = 'Sell'
labels.loc[troughs, 'Label'] = 'Buy'

# Merging the FFT features and labels with the Forex data
final_dataset = forex_data.join(fft_features).join(labels)

# Data Preprocessing
final_dataset.fillna({'Frequency': 0, 'Amplitude': 0, 'DaysPerCycle': 0}, inplace=True)
final_dataset.dropna(subset=['Label'], inplace=True)
final_dataset['Label'] = final_dataset['Label'].map({'Buy': 1, 'Sell': 0})
X = final_dataset[['Frequency', 'Amplitude', 'DaysPerCycle']]
y = final_dataset['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = logreg.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Backtesting the trading strategy
backtest_dataset = final_dataset.copy()
backtest_dataset['PredictedLabel'] = logreg.predict(scaler.transform(backtest_dataset[['Frequency', 'Amplitude', 'DaysPerCycle']]))

cash = 10000  # Starting cash
shares = 0    # Number of shares held
trade_log = []  # Log of trades

for index, row in backtest_dataset.iterrows():
    if row['PredictedLabel'] == 1 and cash >= row['Open']:
        num_shares_to_buy = int(cash / row['Open'])
        shares += num_shares_to_buy
        cash -= num_shares_to_buy * row['Open']
        trade_log.append(f"Buy {num_shares_to_buy} shares at {row['Open']} on {row['Date']}")
    elif row['PredictedLabel'] == 0 and shares > 0:
        cash += shares * row['Open']
        trade_log.append(f"Sell {shares} shares at {row['Open']} on {row['Date']}")
        shares = 0

final_portfolio_value = cash + shares * backtest_dataset.iloc[-1]['Open']
print(trade_log[:10])  # Display first 10 trades
print(f"Final Portfolio Value: {final_portfolio_value}")
