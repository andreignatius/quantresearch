import os
import pandas as pd
import numpy as np
import talib

# Load the Forex data
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, '../backtest/data/USD_JPY_YF.csv')
forex_data = pd.read_csv(file_path)

### TECHNICAL INDICATORS

# Aroon
def calculate_aroon(df):
    df['Aroon_Up'], df['Aroon_Down'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
    return df

# Average True Range
def calculate_atr(df):
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    return df

# Awesome Oscillator
def calculate_awesome_oscillator(df):
    # Calculate median price
    df['median_price'] = (df['High'] + df['Low']) / 2
    # Calculate 5-period SMA of median price
    df['SMA_5'] = df['median_price'].rolling(window=5).mean()
    # Calculate 34-period SMA of median price
    df['SMA_34'] = df['median_price'].rolling(window=34).mean()
    # Calculate Awesome Oscillator
    df['Awesome_Oscillator'] = df['SMA_5'] - df['SMA_34']
    # Drop intermediate columns
    df.drop(['median_price', 'SMA_5', 'SMA_34'], axis=1, inplace=True)
    return df

# Bollinger Bands
def calculate_bollinger_bands(df):
    df['BBand_Upper'], df['BBand_Middle'], df['BBand_Lower'] = talib.BBANDS(df['Close'], timeperiod=20)
    return df

# Bollinger Bandwidth
def calculate_bollinger_bandwidth(df):
    df['Bollinger_Bandwidth'] = (df['BBand_Upper'] - df['BBand_Lower']) / df['BBand_Middle']
    return df

# Bollinger %B
def calculate_bollinger_percent_b(df):
    df['Bollinger_PercentB'] = (df['Close'] - df['BBand_Lower']) / (df['BBand_Upper'] - df['BBand_Lower'])
    return df

# Chaikin Volatility
def calculate_chaikin_volatility(df):
    period = 10
    high = df['High']
    low = df['Low']
    ema_high = high.ewm(span=period, min_periods=period).mean()
    ema_low = low.ewm(span=period, min_periods=period).mean()
    sma_diff = (high - low).rolling(window=period).mean()
    df['chaikin_vol'] = ((ema_high - ema_low) / sma_diff) * np.sqrt(period)
    return df

# Commodity Channel Index (CCI)
def calculate_cci(df):
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=20)
    return df

# Detrended Price Oscillator (DPO)
def calculate_dpo_5(df):
    period = 5
    # Calculate the moving average
    df['Moving_Average'] = df['Close'].rolling(window=period).mean().shift(period // 2 + 1)
    # Calculate Detrended Price Oscillator (DPO)
    df['DPO'] = df['Close'] - df['Moving_Average']
    return df

# Donchian Channel
def calculate_donchian_channel(df, window=20):
    high = df['High']
    low = df['Low']
    df['DC_High'] = high.rolling(window=window).max()
    df['DC_Low'] = low.rolling(window=window).min()
    return df

# Ease of Movement
def calculate_ease_of_movement(df):
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    distance_moved = ((high + low) / 2) - ((high.shift() + low.shift()) / 2)
    box_ratio = volume / (high - low)
    ease_of_movement = distance_moved / box_ratio
    df['EoM'] = ease_of_movement
    return df

# Exponential Moving Average
def calculate_exponential_moving_average(df, window=14):
    df['EMA'] = df['Close'].ewm(span=window, min_periods=0, adjust=False).mean()
    return df

# Ichimoku
def calculate_ichimoku(df, tenkan_period=9, kijun_period=26):
    high = df['High']
    low = df['Low']
    df['Ichimoku_Tenkan'] = (high.rolling(window=tenkan_period).max() + low.rolling(window=tenkan_period).min()) / 2
    df['Ichimoku_Kijun'] = (high.rolling(window=kijun_period).max() + low.rolling(window=kijun_period).min()) / 2
    return df

# KDJ
def calculate_kdj(df, window=14, k_window=3, d_window=3):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    
    rsv = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k_value = rsv.ewm(span=k_window, min_periods=0, adjust=False).mean()
    d_value = k_value.ewm(span=d_window, min_periods=0, adjust=False).mean()
    
    df['KDJ_K'] = k_value
    df['KDJ_D'] = d_value
    
    return df

# Directional Movement Index (DMI)
def calculate_dmi(df):
    df['+DI'], df['-DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14), talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    return df

# Moving Average Convergence Divergence (MACD)
def calculate_macd(df):
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    return df

# Calculate moving average of last 5 days
def calculate_moving_average_5(df):
    window_size = 5
    df['Moving_Average'] = df['Close'].rolling(window=window_size).mean()
    return df

# Relative Strength Index (RSI)
def calculate_rsi(df):
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    return df

# Keltner Channel
def calculate_keltner_channel(df, period=20, multiplier=2):
    df['KC_middle'] = talib.SMA(df['Close'], timeperiod=period)
    df['KC_upper'] = df['KC_middle'] + multiplier * talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
    df['KC_lower'] = df['KC_middle'] - multiplier * talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
    return df

# MACD
def calculate_macd(df):
    macd, signal, _ = talib.MACD(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    return df

# Momentum
def calculate_momentum(df, period=14):
    df['Momentum'] = df['Close'].diff(period)
    return df

# Money Flow Index (MFI)
def calculate_mfi(df, period=14):
    df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=period)
    return df

# Parabolic SAR
def calculate_parabolic_sar(df):
    df['Parabolic_SAR'] = talib.SAR(df['High'], df['Low'])
    return df

# Pivot Point
def calculate_pivot_point(df):
    df['Pivot_Point'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    return df

# Rate of Change
def calculate_roc(df, period=12):
    df['ROC'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
    return df

# RCI - Relative Comparison Index
def calculate_rci(df, period=9):
    df['RCI'] = talib.ROCP(df['Close'], timeperiod=period)
    return df

# Standard Deviation
def calculate_standard_deviation(df):
    window_size = 20  # You can adjust the window size as needed
    df['Standard_Deviation'] = df['Close'].rolling(window=window_size).std()
    return df

# Stochastic
def calculate_stochastic(df):
    df['%K'], df['%D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    return df

# SuperTrend
def calculate_super_trend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=period)
    
    df['Upper_Band'] = hl2 + (multiplier * df['ATR'])
    df['Lower_Band'] = hl2 - (multiplier * df['ATR'])
    
    df['Trend_Up'] = np.nan
    df['Trend_Down'] = np.nan
    
    trend_up = True
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Upper_Band'].iloc[i - 1]:
            trend_up = True
            df.at[i, 'Trend_Up'] = df['Upper_Band'].iloc[i]
            df.at[i, 'Trend_Down'] = np.nan
        elif df['Close'].iloc[i] < df['Lower_Band'].iloc[i - 1]:
            trend_up = False
            df.at[i, 'Trend_Down'] = df['Lower_Band'].iloc[i]
            df.at[i, 'Trend_Up'] = np.nan
        else:
            if trend_up:
                df.at[i, 'Trend_Up'] = df['Upper_Band'].iloc[i]
                df.at[i, 'Trend_Down'] = np.nan
            else:
                df.at[i, 'Trend_Down'] = df['Lower_Band'].iloc[i]
                df.at[i, 'Trend_Up'] = np.nan
    
    return df

# Weighted Moving Average
def calculate_weighted_moving_average(df):
    window_size = 10  # You can adjust the window size as needed
    weights = np.arange(1, window_size + 1)
    df['Weighted_MA'] = df['Close'].rolling(window=window_size).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    return df

# Usage
forex_data = calculate_aroon(forex_data)
forex_data = calculate_atr(forex_data)
forex_data = calculate_awesome_oscillator(forex_data)
forex_data = calculate_bollinger_bands(forex_data)
forex_data = calculate_bollinger_percent_b(forex_data)
forex_data = calculate_bollinger_bandwidth(forex_data)
forex_data = calculate_chaikin_volatility(forex_data)
forex_data = calculate_cci(forex_data)
forex_data = calculate_dpo_5(forex_data)
forex_data = calculate_dmi(forex_data)
forex_data = calculate_donchian_channel(forex_data)
forex_data = calculate_ease_of_movement(forex_data)
forex_data = calculate_exponential_moving_average(forex_data)
forex_data = calculate_ichimoku(forex_data)
forex_data = calculate_kdj(forex_data)
forex_data = calculate_donchian_channel(forex_data)
# forex_data = calculate_ease_of_movement(forex_data)
forex_data = calculate_exponential_moving_average(forex_data)
forex_data = calculate_kdj(forex_data)
forex_data = calculate_macd(forex_data)
forex_data = calculate_moving_average_5(forex_data)
forex_data = calculate_rsi(forex_data)
forex_data = calculate_keltner_channel(forex_data)
forex_data = calculate_macd(forex_data)
forex_data = calculate_momentum(forex_data)
# forex_data = calculate_mfi(forex_data)
forex_data = calculate_parabolic_sar(forex_data)
forex_data = calculate_pivot_point(forex_data)
forex_data = calculate_roc(forex_data)
forex_data = calculate_rci(forex_data)
forex_data = calculate_standard_deviation(forex_data)
forex_data = calculate_stochastic(forex_data)
forex_data = calculate_super_trend(forex_data)
forex_data = calculate_weighted_moving_average(forex_data)

# Outputs
pd.set_option('display.max_columns', None)
# print(forex_data.head())
# print(forex_data.tail())

# Save to CSV
forex_data.to_csv("technical_indicators.csv")
