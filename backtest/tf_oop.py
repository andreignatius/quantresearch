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

# from base_model import BaseModel

# # Load the Forex data
# file_path = 'USD_JPY.csv'
# model = BaseModel(file_path, 'tf_nn')

# data = model.load_preprocess_data()
# model.train()
# print("GOING TO PREDICT")
# predicted_categories = model.predict()

from ml_models.tf_nn_model import TF_NN_Model
# Load the Forex data
file_path = 'data/USD_JPY.csv'
model = TF_NN_Model(file_path)

model.load_preprocess_data()
model.train()
data = model.retrieve_test_set()
print("GOING TO PREDICT")
predicted_categories = model.predict()

# Backtesting with stop-loss and take-profit
stop_loss_threshold = 0.05  # 5% drop from buying price
take_profit_threshold = 0.05  # 5% rise from buying price
cash = 10000  # Starting cash
starting_cash = cash
trading_lot = 2000
shares = 0    # Number of shares held
trade_log = []  # Log of trades
buy_price = None
for index, (row, prediction) in enumerate(zip(data.iterrows(), predicted_categories)):
    # print("row: ", row)
    # print("row[0]: ", row[0])
    # print("row[1]: ", row[1])
    current_price = row[1]['Open']

    if shares > 0:
        change_percentage = (current_price - buy_price) / buy_price
        if change_percentage <= -stop_loss_threshold:
            cash += shares * current_price
            trade_log.append(f"Sell {shares} shares at {current_price} on {row['Date']} (Stop-loss triggered)")
            shares = 0
            continue

    if prediction == 'Buy' and cash >= trading_lot and ( buy_price is None or (buy_price is not None and ( current_price < buy_price * 0.99 or current_price > buy_price * 1.01) ) ):
        print("current_price: ", current_price)
        print("buy_price: ", buy_price)
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
final_portfolio_value = cash + (shares * data.iloc[-1]['Open'])

# Output
print(trade_log)
print("num trades: ", len(trade_log))
print(f"Final Portfolio Value: {final_portfolio_value}")

pnl_per_trade = ( final_portfolio_value - starting_cash ) / len(trade_log)
print("PnL per trade: ", pnl_per_trade)

model.evaluate()
