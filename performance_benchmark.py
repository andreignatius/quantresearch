import pandas as pd
import numpy as np
import os
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

# Load the Forex data
current_directory = os.getcwd()
file_path = file_path = os.path.join(current_directory, 'backtest', 'data', 'USD_JPY.csv')
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

# local_trough = float('inf')
# local_peak = -float('inf')
checkpoint_date_bottom = None  # Initialize to a sensible default or first date
checkpoint_date_top = None  # Initialize to a sensible default or first date

for index, row in final_dataset_with_new_features.iterrows():
    print("index: ", index, "row: ", row)
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
y = final_dataset_with_new_features['Label']

# Splitting the dataset and standardizing features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("y_train value counts: ", y_train.value_counts())
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
# logreg = LogisticRegression()
logreg = LogisticRegression(class_weight='balanced')
logreg.fit(X_train_scaled, y_train)

# evaluate log reg model
y_pred = logreg.predict(X_test_scaled)
print("classification report: ", classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))


# # Note: You'll need to adapt this for multi-class or choose a binary subset of your data.
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
# roc_auc = auc(fpr, tpr)


# scores = cross_val_score(logreg, X_test_scaled, y, cv=5)
# print(scores)

# print(logreg.coef_)


# Predicting labels using the logistic regression model
final_dataset_with_new_features['PredictedLabel'] = logreg.predict(scaler.transform(X))

# Backtesting with stop-loss and take-profit
stop_loss_threshold = 0.05  # 5% drop from buying price
take_profit_threshold = 0.05  # 5% rise from buying price
cash = 10000  # Starting cash
trading_lot = 2000
shares = 0    # Number of shares held
trade_log = []  # Log of trades

print("final_dataset_with_new_features: ", final_dataset_with_new_features)
print("***")
print("\n")

for index, row in final_dataset_with_new_features.iterrows():
    # print("index: ", index, "row: ", row)
    current_price = row['Open']

    # if shares > 0:
    #     change_percentage = (current_price - buy_price) / buy_price
    #     if change_percentage <= -stop_loss_threshold or change_percentage >= take_profit_threshold:
    #         cash += shares * current_price
    #         trade_log.append(f"Sell {shares} shares at {current_price} on {row['Date']} (Stop-loss/Take-profit triggered)")
    #         shares = 0
    #         continue

    # Model-based trading decisions
    if row['Label'] == 'Buy' and cash >= trading_lot:  # Buy signal
        # num_shares_to_buy = int(cash / current_price)
        num_shares_to_buy = int(trading_lot / current_price)
        shares += num_shares_to_buy
        cash -= num_shares_to_buy * current_price
        buy_price = current_price
        trade_log.append(f"Buy {num_shares_to_buy} shares at {current_price} on {row['Date']}")
        print(f"ACTION : Buying {num_shares_to_buy} shares at {current_price} on {row['Date']}")
    elif row['Label'] == 'Sell' and shares > 0:  # Sell signal
        cash += shares * current_price
        print("CHECK : cash amt ", cash)
        # num_shares_to_sell = int(trading_lot / current_price)
        # cash += 
        trade_log.append(f"Sell {shares} shares at {current_price} on {row['Date']} (Model signal)")
        print(f"ACTION : Selling {shares} shares at {current_price} on {row['Date']} (Model signal)")
        shares = 0

# Calculate final portfolio value
final_portfolio_value = cash + shares * final_dataset_with_new_features.iloc[-1]['Open']

# Output
for log in trade_log:
    print(log)
print(f"Final Portfolio Value: {final_portfolio_value}")
