import numpy as np

class TradingStrategy:
    def __init__(self, model, data, start_cash=10000, trading_lot=2500, stop_loss_threshold=0.05, take_profit_threshold=0.05, leverage_factor=3, annual_interest_rate=0.03, margin_call_threshold=0.5):
        self.model = model
        self.data = data
        self.cash = start_cash
        self.starting_cash = start_cash
        self.trading_lot = trading_lot
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.leverage_factor = leverage_factor
        self.annual_interest_rate = annual_interest_rate
        self.margin_call_threshold = margin_call_threshold * start_cash
        self.trade_log = []
        self.buy_price = None
        self.jpy_inventory = 0
        self.daily_return_factors = []  # To store daily return factors for compounded return calculation

    def execute_trades(self):
        predicted_categories = self.model.predict()
        for index, (row, prediction) in enumerate(zip(self.data.iterrows(), predicted_categories)):
            current_price = row[1]['Open']

            if self.jpy_inventory > 0:
                change_percentage = (current_price - self.buy_price) / self.buy_price
                self.daily_return_factors.append(1 + (change_percentage * self.leverage_factor))

                # Check for stop-loss or take-profit
                # if change_percentage <= -self.stop_loss_threshold or change_percentage >= self.take_profit_threshold:
                if change_percentage <= -self.stop_loss_threshold:
                    print("STOP LOSS!!! (should not happen)")
                    self._sell_jpy(current_price, row[1]['Date'])

            if prediction == 'Sell' and self.cash >= self.trading_lot:
                self._buy_jpy(current_price, row[1]['Date'])

            # Check for margin call
            if self._check_margin_call():
                print("MARGIN CALL!!! (should not happen)")
                self._sell_jpy(current_price, row[1]['Date'], forced=True)

        self._apply_interest_charge()  # Apply interest charge at the end of trading

    def _buy_jpy(self, rate, date):
        jpy_bought = int((self.trading_lot * self.leverage_factor) / rate)
        self.jpy_inventory += jpy_bought
        self.cash -= self.trading_lot
        self.buy_price = rate
        self.trade_log.append(f"Buy {jpy_bought} JPY at {rate} on {date}")

    def _sell_jpy(self, rate, date, forced=False):
        if self.jpy_inventory <= 0:
            return

        jpy_convert_to_usd = self.jpy_inventory / rate
        self.cash += jpy_convert_to_usd
        sell_reason = "Stop-loss triggered" if not forced else "Margin call"
        self.trade_log.append(f"Sell {self.jpy_inventory} JPY at {rate} on {date} ({sell_reason})")
        self.jpy_inventory = 0

    def _check_margin_call(self):
        if self.jpy_inventory > 0:
            current_value = self.jpy_inventory / self.buy_price
            if current_value < self.margin_call_threshold:
                return True
        return False

    def _apply_interest_charge(self):
        days_held = len(self.daily_return_factors)
        daily_interest_rate = (1 + self.annual_interest_rate) ** (1/365) - 1
        interest_charge = self.trading_lot * self.leverage_factor * daily_interest_rate * days_held
        self.cash -= interest_charge

    def evaluate_performance(self):
        compounded_return = np.prod(self.daily_return_factors)
        final_portfolio_value = self.cash + (self.jpy_inventory * compounded_return)
        pnl_per_trade = (final_portfolio_value - self.starting_cash) / len(self.trade_log) if self.trade_log else 0
        return {
            'Final Portfolio Value': final_portfolio_value,
            'Number of Trades': len(self.trade_log),
            'Profit/Loss per Trade': pnl_per_trade,
            'Trade Log': self.trade_log
        }
