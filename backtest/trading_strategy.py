
class TradingStrategy:
    def __init__(self, model, data, start_cash=10000, trading_lot=2000, stop_loss_threshold=0.05, take_profit_threshold=0.05):
        self.model = model
        self.data = data
        self.cash = start_cash
        self.starting_cash = start_cash
        self.trading_lot = trading_lot
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.trade_log = []
        self.buy_price = None
        self.shares = 0

    def execute_trades(self):
        predicted_categories = self.model.predict()

        for index, (row, prediction) in enumerate(zip(self.data.iterrows(), predicted_categories)):
            # print("row: ", row)
            # print("row[0]: ", row[0])
            # print("row[1]: ", row[1])
            current_price = row[1]['Open']

            if self.shares > 0:
                change_percentage = (current_price - self.buy_price) / self.buy_price
                if change_percentage <= -self.stop_loss_threshold:
                    self.cash += self.shares * current_price
                    self.trade_log.append(f"Sell {self.shares} shares at {current_price} on {row[1]['Date']} (Stop-loss triggered)")
                    self.shares = 0
                    continue

            if prediction == 'Buy' and self.cash >= self.trading_lot and ( self.buy_price is None or (self.buy_price is not None and ( current_price < self.buy_price * 0.99 or current_price > self.buy_price * 1.01) ) ):
                print("current_price: ", current_price)
                print("buy_price: ", self.buy_price)
                num_shares_to_buy = int(self.trading_lot / current_price)
                self.shares += num_shares_to_buy
                self.cash -= num_shares_to_buy * current_price
                self.buy_price = current_price
                self.trade_log.append(f"Buy {num_shares_to_buy} shares at {current_price} on {row[1]['Date']}")
            elif prediction == 'Sell' and self.shares > 0:
                self.cash += self.shares * current_price
                self.trade_log.append(f"Sell {self.shares} shares at {current_price} on {row[1]['Date']}")
                self.shares = 0

    def evaluate_performance(self):
        final_portfolio_value = self.cash + (self.shares * self.data.iloc[-1]['Open'])
        pnl_per_trade = (final_portfolio_value - self.starting_cash) / len(self.trade_log) if self.trade_log else 0
        return {
            'Final Portfolio Value': final_portfolio_value,
            'Number of Trades': len(self.trade_log),
            'Profit/Loss per Trade': pnl_per_trade,
            'Trade Log': self.trade_log
        }
