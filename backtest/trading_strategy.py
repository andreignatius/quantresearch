
class TradingStrategy:

    def __init__(self, model, data, start_cash=10000, trading_lot=2500, stop_loss_threshold=0.05, take_profit_threshold=0.05, leverage_factor=3):
        self.model = model
        self.data = data
        self.cash = start_cash
        self.starting_cash = start_cash
        self.trading_lot = trading_lot * leverage_factor  # Adjusted for leverage
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.leverage_factor = leverage_factor
        self.trade_log = []
        self.buy_price = None
        self.jpy_inventory = 0

    def execute_trades(self):
        predicted_categories = self.model.predict()
        print("CASH000: ", self.cash)
        for index, (row, prediction) in enumerate(zip(self.data.iterrows(), predicted_categories)):
            # print("row: ", row)
            # print("row[0]: ", row[0])
            # print("row[1]: ", row[1])
            usd_jpy_spot_rate = row[1]['Open']

            # if self.jpy_inventory > 0:
            #     change_percentage = (usd_jpy_spot_rate - self.buy_price) / self.buy_price
            #     if change_percentage <= -self.stop_loss_threshold:
            #         self.cash += self.jpy_inventory * usd_jpy_spot_rate
            #         self.trade_log.append(f"Sell {self.jpy_inventory} shares at {usd_jpy_spot_rate} on {row[1]['Date']} (Stop-loss triggered)")
            #         self.jpy_inventory = 0
            #         continue

            if prediction == 'Sell' and self.cash >= self.trading_lot and ( self.buy_price is None or (self.buy_price is not None and ( usd_jpy_spot_rate < self.buy_price * 0.99 or usd_jpy_spot_rate > self.buy_price * 1.01) ) ):
                print("usd_jpy_spot_rate: ", usd_jpy_spot_rate)
                print("buy_price: ", self.buy_price)
                jpy_bought = int(self.trading_lot * usd_jpy_spot_rate)
                self.jpy_inventory += jpy_bought
                self.cash -= self.trading_lot
                self.buy_price = usd_jpy_spot_rate
                self.trade_log.append(f"Buy {jpy_bought} JPY at {usd_jpy_spot_rate} on {row[1]['Date']}")
            elif prediction == 'Buy' and self.jpy_inventory > 0:
                jpy_convert_to_usd = self.jpy_inventory / usd_jpy_spot_rate
                print("jpy_convert_to_usd: ", jpy_convert_to_usd)
                print("cash000: ", self.cash)
                self.cash += jpy_convert_to_usd
                print("cash111: ", self.cash)
                self.trade_log.append(f"Sell {self.jpy_inventory} JPY at {usd_jpy_spot_rate} on {row[1]['Date']}")
                self.jpy_inventory = 0

    def evaluate_performance(self):
        final_usd_jpy_spot_rate = self.data.iloc[-1]['Open']
        final_portfolio_value = self.cash + (self.jpy_inventory / final_usd_jpy_spot_rate)
        print("cash: ", self.cash)
        print("shares: ", self.jpy_inventory)
        pnl_per_trade = (final_portfolio_value - self.starting_cash) / len(self.trade_log) if self.trade_log else 0
        return {
            'Final Portfolio Value': final_portfolio_value,
            'Number of Trades': len(self.trade_log),
            'Profit/Loss per Trade': pnl_per_trade,
            'Trade Log': self.trade_log
        }
