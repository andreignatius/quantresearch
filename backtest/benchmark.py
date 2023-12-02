from ml_models.base_model import BaseModel
import os
import pandas as pd
from datetime import datetime
from trading_strategy import TradingStrategy


# # Load the Forex data
# # current_directory = os.getcwd()
# current_directory = os.path.dirname(__file__)
# file_path = os.path.join(current_directory, 'data/USD_JPY.csv')
# # file_path = 'data/USD_JPY_bbg.csv'
# model = BaseModel(file_path)

# model.load_preprocess_data()
# data = model.retrieve_test_set()




# Load the Forex data
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data/USD_JPY_YF.csv')
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
data.sort_values('Date', inplace=True)      # Sort by 'Date'


def rolling_window_train_predict(data, start_year, end_year, train_duration, test_duration):
    trade_logs = []
    final_portfolio_values = []

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
        model = BaseModel(file_path, train_start, train_end, test_start, test_end)
        model.load_preprocess_data()
        # model.train()
        model.train_test_split_time_series()
        data = model.retrieve_test_set()
        # predicted_categories = model.predict()

        # Perform backtesting, log trades, calculate final portfolio value
        # ... [Your backtesting logic here] ...
        # Backtesting with stop-loss and take-profit
        # Instantiate the TradingStrategy class
        # Backtesting with stop-loss and take-profit
        stop_loss_threshold = 0.05  # 5% drop from buying price
        take_profit_threshold = 0.05  # 5% rise from buying price
        cash = 10000  # Starting cash
        starting_cash = cash
        trading_lot = 2000
        shares = 0    # Number of shares held
        trade_log = []  # Log of trades
        buy_price = None
        for index, row in data.iterrows():
            print("index: ", index, "row: ", row)
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
                trade_log.append(
                    f"Buy {num_shares_to_buy} shares at {current_price} on {row['Date']}")
                print(
                    f"ACTION : Buying {num_shares_to_buy} shares at {current_price} on {row['Date']}")
            elif row['Label'] == 'Sell' and shares > 0:  # Sell signal
                cash += shares * current_price
                print("CHECK : cash amt ", cash)
                # num_shares_to_sell = int(trading_lot / current_price)
                # cash +=
                trade_log.append(
                    f"Sell {shares} shares at {current_price} on {row['Date']} (Model signal)")
                print(
                    f"ACTION : Selling {shares} shares at {current_price} on {row['Date']} (Model signal)")
                shares = 0

        # Calculate final portfolio value
        final_portfolio_value = cash + shares * \
            data.iloc[-1]['Open']

        # Output
        for log in trade_log:
            print(log)
        print(f"Final Portfolio Value: {final_portfolio_value}")

        # trade_log = trading_results['Trade Log']
        # final_portfolio_value = trading_results['Final Portfolio Value']
        # pnl_per_trade = trading_results['Profit/Loss per Trade']

        # Output
        print(trade_log)
        print("num trades: ", len(trade_log))
        print(f"Final Portfolio Value: {final_portfolio_value}")

        pnl_per_trade = ( final_portfolio_value - starting_cash ) / len(trade_log)
        print("PnL per trade: ", pnl_per_trade)

        # Collect results
        trade_logs.append(trade_log)
        final_portfolio_values.append(final_portfolio_value)

        # Move to the next window
        current_date += pd.DateOffset(months=6)
        train_duration += 6

    return trade_logs, final_portfolio_values


# Apply the rolling window approach
trade_logs, final_values = rolling_window_train_predict(data, 2013, 2023, 12, 6)  # 12 months training, 6 months testing

pnl_per_quarter = [x - 10000 for x in final_values]

print("final trade_logs: ", trade_logs)
print("final_values: ", final_values)
print("pnl_per_quarter: ", pnl_per_quarter)
print("final_pnl: ", sum(pnl_per_quarter) )
percentage_returns = ( sum(pnl_per_quarter) / len(pnl_per_quarter) ) / 10000 * 100
print("percentage_returns: ", percentage_returns)
