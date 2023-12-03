
class TradingStrategy:

    def __init__(self, model, data, start_cash=10000, trading_lot=2500, stop_loss_threshold=0.05, take_profit_threshold=0.05, leverage_factor=3, margin_call_threshold=0.5):
        self.model = model
        self.data = data
        self.cash = start_cash
        self.margin_requirement = start_cash * margin_call_threshold
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
            usd_jpy_spot_rate = row[1]['Open']
            current_date = row[1]['Date']

            is_stop_loss_triggered = self._check_stop_loss(usd_jpy_spot_rate, current_date)

            if is_stop_loss_triggered:
                continue

            if prediction == 'Sell' and self.cash >= self.trading_lot and ( self.buy_price is None or (self.buy_price is not None and ( usd_jpy_spot_rate < self.buy_price * 0.99 or usd_jpy_spot_rate > self.buy_price * 1.01) ) ):
                self._buy_jpy(usd_jpy_spot_rate, current_date)
            elif prediction == 'Buy' and self.jpy_inventory > 0:
                self._sell_jpy(usd_jpy_spot_rate, current_date)

            if self._check_margin_call(usd_jpy_spot_rate):
                print("MARGIN CALL!!! this should not happen!")
                self._sell_jpy(usd_jpy_spot_rate, current_date)

    def _buy_jpy(self, rate, date):
        jpy_bought = int(self.trading_lot * rate)
        self.jpy_inventory += jpy_bought
        self.cash -= self.trading_lot
        self.buy_price = rate
        self.trade_log.append(f"Buy {jpy_bought} JPY at {rate} on {date}")

    def _sell_jpy(self, rate, date, forced=False):
        if self.jpy_inventory <= 0:
            return

        jpy_convert_to_usd = self.jpy_inventory / rate
        self.cash += jpy_convert_to_usd
        sell_reason = "Model predicted sell" if not forced else "Margin call / stop-loss triggered"
        self.trade_log.append(f"Sell {self.jpy_inventory} JPY at {rate} on {date} ({sell_reason})")
        self.jpy_inventory = 0

    def _compute_mtm(self, usd_jpy_spot_rate):
        return self.cash + ( self.jpy_inventory / usd_jpy_spot_rate )

    def _check_stop_loss(self, usd_jpy_spot_rate, date):
        if self.jpy_inventory > 0:
            change_percentage = (usd_jpy_spot_rate - self.buy_price) / self.buy_price
            if change_percentage <= -self.stop_loss_threshold:
                self._sell_jpy(usd_jpy_spot_rate, date, forced=True)
                return True
        return False

    def _check_margin_call(self, usd_jpy_spot_rate):
        if self.jpy_inventory > 0:
            if self._compute_mtm(usd_jpy_spot_rate) < self.margin_requirement:
                return True
        return False

    # def _apply_interest_charge(self):
    #     days_held = len(self.daily_return_factors)
    #     daily_interest_rate = (1 + self.annual_interest_rate) ** (1/365) - 1
    #     interest_charge = self.trading_lot * self.leverage_factor * daily_interest_rate * days_held
    #     self.cash -= interest_charge

    def evaluate_performance(self):
        final_usd_jpy_spot_rate = self.data.iloc[-1]['Open']
        # final_portfolio_value = self.cash + (self.jpy_inventory / final_usd_jpy_spot_rate)
        final_portfolio_value = self._compute_mtm(final_usd_jpy_spot_rate)
        print("cash: ", self.cash)
        print("shares: ", self.jpy_inventory)
        pnl_per_trade = (final_portfolio_value - self.starting_cash) / len(self.trade_log) if self.trade_log else 0
        return {
            'Final Portfolio Value': final_portfolio_value,
            'Number of Trades': len(self.trade_log),
            'Profit/Loss per Trade': pnl_per_trade,
            'Trade Log': self.trade_log
        }
