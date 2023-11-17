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

# Calculating Moving Averages and RSI manually
def calculate_rsi(data, window=14):
    """ Calculate the Relative Strength Index (RSI) for a given dataset and window """
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

short_window = 5
long_window = 20
rsi_period = 14
forex_data['Short_Moving_Avg'] = forex_data['Close'].rolling(window=short_window).mean()
forex_data['Long_Moving_Avg'] = forex_data['Close'].rolling(window=long_window).mean()
forex_data['RSI'] = calculate_rsi(forex_data, window=rsi_period)

# Merging the new features into the final dataset
final_dataset_with_new_features = final_dataset.join(forex_data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI']])

# Data Preprocessing
final_dataset_with_new_features.dropna(inplace=True)
X = final_dataset_with_new_features[['Frequency', 'Amplitude', 'DaysPerCycle', 'Short_Moving_Avg', 'Long_Moving_Avg', 'RSI']]
y = final_dataset_with_new_features['Label']

# Splitting the dataset and standardizing features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Predicting labels using the logistic regression model
final_dataset_with_new_features['PredictedLabel'] = logreg.predict(scaler.transform(X))

# Backtesting with stop-loss and take-profit
stop_loss_threshold = 0.05  # 5% drop from buying price
take_profit_threshold = 0.05  # 5% rise from buying price
cash = 10000  # Starting cash
shares = 0    # Number of shares held
trade_log = []  # Log of trades

for index, row in final_dataset_with_new_features.iterrows():
    current_price = row['Open']
    if shares > 0:
        change_percentage = (current_price - buy_price) / buy_price
        if change_percentage <= -stop_loss_threshold or change_percentage >= take_profit_threshold:
            cash += shares * current_price
            trade_log.append(f"Sell {shares} shares at {current_price} on {row['Date']} (Stop-loss/Take-profit triggered)")
            shares = 0
            continue

    # Model-based trading decisions
    if row['PredictedLabel'] == 1 and cash >= current_price:  # Buy signal
        num_shares_to_buy = int(cash / current_price)
        shares += num_shares_to_buy
        cash -= num_shares_to_buy * current_price
        buy_price = current_price
        trade_log.append(f"Buy {num_shares_to_buy} shares at {current_price} on {row['Date']}")
    elif row['PredictedLabel'] == 0 and shares > 0:  # Sell signal
        cash += shares * current_price
        trade_log.append(f"Sell {shares} shares at {current_price} on {row['Date']} (Model signal)")
        shares = 0

# Calculate final portfolio value
final_portfolio_value = cash + shares * final_dataset_with_new_features.iloc[-1]['Open']

# Output
print(trade_log[:10])  # Display first 10 trades
print(f"Final Portfolio Value: {final_portfolio_value}")
