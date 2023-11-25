from ml_models.base_model import BaseModel

# Load the Forex data
file_path = 'data/USD_JPY.csv'
# file_path = 'data/USD_JPY_bbg.csv'
model = BaseModel(file_path)

model.load_preprocess_data()

data = model.retrieve_test_set()

# Backtesting with stop-loss and take-profit
stop_loss_threshold = 0.05  # 5% drop from buying price
take_profit_threshold = 0.05  # 5% rise from buying price
cash = 10000  # Starting cash
starting_cash = cash
trading_lot = 2000
shares = 0    # Number of shares held
trade_log = []  # Log of trades
buy_price = None

print("check data: ", data)

for index, row in data.iterrows():
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
