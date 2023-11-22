from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

import random

# Load the Forex data
file_path = 'backtest/data/USD_JPY.csv'
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
mid_trend = [i for i in range(len(close_prices))]
print("check mid_trend: ", mid_trend)
for peak in peaks:
    mid_trend.remove(peak)
for trough in troughs:
    mid_trend.remove(trough)
print("check mid_trend1: ", mid_trend)
labels = pd.DataFrame(index=range(len(close_prices)), columns=['Label'])
print("labels: ", labels)
for label in labels:
    print("c: ", label)
labels.loc[peaks, 'Label'] = 'Sell'
labels.loc[troughs, 'Label'] = 'Buy'
labels.loc[mid_trend, 'Label'] = 'Hold'

print("fft_features: ", fft_features)
print("forex_data: ", forex_data)
print("labels: ", labels)
# Merging the FFT features and labels with the Forex data
# final_dataset = forex_data.join(fft_features).join(labels)
final_dataset = forex_data.join(labels)
print("check final_dataset: ", final_dataset)

print("***")
print("\n")

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
print("forex_data1: ", forex_data)
# Merging the new features into the final dataset
final_dataset_with_new_features = final_dataset.join(forex_data[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI']])
print("check final_dataset_with_new_features: ", final_dataset_with_new_features)

# Data Preprocessing
final_dataset_with_new_features.dropna(inplace=True)

checkpoint_date_bottom = None  # Initialize to a sensible default or first date
checkpoint_date_top = None  # Initialize to a sensible default or first date

for index, row in final_dataset_with_new_features.iterrows():
    # print("index: ", index, "row: ", row)
    current_price = row['Open']

    today_date = pd.to_datetime(row['Date'])

    if row['Label'] == 'Buy':
        checkpoint_date_bottom = today_date
        print("CHECK UPDATE local_trough checkpoint_date_bottom: ", today_date)
    if row['Label'] == 'Sell':
        checkpoint_date_top = today_date
        print("CHECK UPDATE local_peak checkpoint_date_top: ", today_date)

    days_since_bottom = (today_date - checkpoint_date_bottom).days if checkpoint_date_bottom else 0
    print("days_since_bottom: ", days_since_bottom)
    days_since_peak = (today_date - checkpoint_date_top).days if checkpoint_date_top else 0
    print("days_since_peak: ", days_since_peak)

    # final_dataset_with_new_features.at[index, 'DaysSincePeakTrough'] = max(days_since_bottom, days_since_peak)
    final_dataset_with_new_features.at[index, 'DaysSincePeak'] = days_since_peak
    final_dataset_with_new_features.at[index, 'DaysSincePeakTrough'] = days_since_bottom
    

final_dataset_with_new_features.to_csv('final_dataset_with_new_features.csv')
# print("check final_dataset_with_new_features: ", final_dataset_with_new_features)
# X = final_dataset_with_new_features[['Frequency', 'Amplitude', 'DaysPerCycle', 'Short_Moving_Avg', 'Long_Moving_Avg', 'RSI']]
X = final_dataset_with_new_features[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI', 'DaysSincePeak', 'DaysSincePeakTrough']]
# X = final_dataset_with_new_features[['Short_Moving_Avg', 'Long_Moving_Avg', 'RSI']]
y = final_dataset_with_new_features['Label']

# Assuming X and y are your features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("y_train value counts: ", y_train.value_counts())
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Assuming X_train_scaled, y_train, X_test_scaled, y_test are already defined

# Convert labels to tensors and apply one-hot encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Convert labels to categorical (one-hot encoding)
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Neural Network architecture
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train_categorical.shape[1], activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train_categorical, epochs=50, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Generate predictions
# predicted_probs = model.predict(X_test_scaled)
predicted_probs = model.predict(X_test_scaled)

# Convert probabilities to class labels
predicted_labels = np.argmax(predicted_probs, axis=1)

# Assuming label_encoder was used to encode y_train
predicted_categories = label_encoder.inverse_transform(predicted_labels)

# Backtesting with stop-loss and take-profit
stop_loss_threshold = 0.05  # 5% drop from buying price
take_profit_threshold = 0.05  # 5% rise from buying price
cash = 10000  # Starting cash
trading_lot = 2000
shares = 0    # Number of shares held
trade_log = []  # Log of trades

for index, (row, prediction) in enumerate(zip(final_dataset_with_new_features.iterrows(), predicted_categories)):
    # print("row: ", row)
    # print("row[0]: ", row[0])
    # print("row[1]: ", row[1])
    current_price = row[1]['Open']
    
    if prediction == 'Buy' and cash >= trading_lot:
        num_shares_to_buy = int(trading_lot / current_price)
        shares += num_shares_to_buy
        cash -= num_shares_to_buy * current_price
        buy_price = current_price
        trade_log.append(f"Buy {num_shares_to_buy} shares at {current_price} on {row[1]['Date']}")
    elif prediction == 'Sell' and shares > 0:
        cash += shares * current_price
        trade_log.append(f"Sell {shares} shares at {current_price} on {row[1]['Date']}")
        shares = 0

# Calculate final portfolio value
final_portfolio_value = cash + (shares * final_dataset_with_new_features.iloc[-1]['Open'])

# Output
print(trade_log)
print(f"Final Portfolio Value: {final_portfolio_value}")
