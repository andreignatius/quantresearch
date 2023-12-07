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

# from base_model import BaseModel
from ml_models.gbt_model import GBTModel
from trading_strategy import TradingStrategy

from datetime import datetime

# Load the Forex data
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data/USD_JPY_YF.csv')
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
data.sort_values('Date', inplace=True)      # Sort by 'Date'


def rolling_window_train_predict(data, start_year, end_year, train_duration, test_duration):
    trade_logs = []
    pnl_logs = []
    final_portfolio_values = []
    interest_costs_total = []
    transaction_costs_total = []

    # Convert 'Date' column to datetime if it's not already
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the data by date to ensure correct time sequence
    data.sort_values('Date', inplace=True)

    start_date = datetime(start_year, 1, 1)
    current_date = datetime(start_year, 1, 1)

    while current_date.year < end_year:
        # Define the start and end of the training period
        # train_start = current_date
        train_start = start_date
        train_end = train_start + pd.DateOffset(months=train_duration)

        # Define the start and end of the testing period
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_duration)

        # Ensure not to exceed the dataset
        if test_end.year > end_year:
            break

        print("train_start: ", train_start)
        print("train_end: ", train_end)
        print("test_start: ", test_start)
        print("test_end: ", test_end)
        # Train and predict
        model = GBTModel(file_path, train_start, train_end, test_start, test_end)
        model.load_preprocess_data()
        model.train()
        data = model.retrieve_test_set()
        # predicted_categories = model.predict()

        # Perform backtesting, log trades, calculate final portfolio value
        # ... [Your backtesting logic here] ...
        # Backtesting with stop-loss and take-profit
        # Instantiate the TradingStrategy class
        trading_strategy = TradingStrategy(model, data)

        # Run the trading strategy with the model's predictions
        trading_strategy.execute_trades()

        # Retrieve results and output
        trading_results = trading_strategy.evaluate_performance()

        trade_log               = trading_results['Trade Log']
        pnl_log                 = trading_results['PNL Log']
        final_portfolio_value   = trading_results['Final Portfolio Value']
        pnl_per_trade           = trading_results['Profit/Loss per Trade']
        interest_costs          = sum(trading_results['Interest Costs'])
        transaction_costs       = trading_results['Transaction Costs']

        print("interest_costs111: ", interest_costs)
        print("transaction_costs111: ", transaction_costs)

        interest_costs_total.append( interest_costs )
        transaction_costs_total.append( transaction_costs )

        # Output
        print(trade_log)
        print("num trades: ", len(trade_log))
        print(f"Final Portfolio Value Before Cost: {final_portfolio_value}")
        final_portfolio_value = final_portfolio_value - ( interest_costs + transaction_costs )
        print(f"Final Portfolio Value After Cost: {final_portfolio_value}")

        # pnl_per_trade = ( final_portfolio_value - starting_cash ) / len(trade_log)
        print("PnL per trade: ", pnl_per_trade)

        # Collect results
        trade_logs.append(trade_log)
        pnl_logs.extend(pnl_log)
        final_portfolio_values.append(final_portfolio_value)

        # Move to the next window
        current_date += pd.DateOffset(months=6)
        train_duration += 6

    return trade_logs, pnl_logs, final_portfolio_values, interest_costs_total, transaction_costs_total


# Apply the rolling window approach
trade_logs, pnl_logs, final_values, interest_costs_total, transaction_costs_total = rolling_window_train_predict(data, 2013, 2023, 12, 6)  # 12 months training, 6 months testing

pnl_per_quarter = [x - 10000 for x in final_values]

print("final trade_logs: ", trade_logs)
print("final pnl_logs: ", pnl_logs)
print("final_values: ", final_values)
print("pnl_per_quarter: ", pnl_per_quarter)
print("final_pnl: ", sum(pnl_per_quarter) )
print("interest_costs: ", interest_costs_total)
print("transaction_costs :", transaction_costs_total)
percentage_returns = ( sum(pnl_per_quarter) / len(pnl_per_quarter) ) / 10000 * 100
print("percentage_returns per period: ", percentage_returns)
print("percentage_returns per annum: ", percentage_returns * 2)

