'''
Trading Indicators
1. SMA
2. EMA
3. RSI
4. MACD
5. stochastic oscillator
6. Bollinger Bands
7. Standard Deviation
8. Average Directional Index
9. Rate of change
10. Momentum
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import yfinance as yf
from scipy.signal import find_peaks
from backtesting import Backtest, Strategy
import talib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../backtest/data/USD_JPY_YF.csv',parse_dates=["Date"]).set_index("Date").drop(columns = "Volume")

def SMA(array, n):
    return talib.SMA(array, n)
def EMA(array, n):
    return talib.EMA(array, n)
def RSI(array, n):
    return talib.RSI(array, n)

class Parameters(Strategy):
    SMA_n = 5
    EMA_n = 5
    RSI_n = 14
    RSI_lower = 40
    RSI_upper = 60

    short_MA = 12
    long_MA = 26
    signal_MA = 9

    fastk_period = 14
    slowk_period = 3
    slowd_period = 3
    lower_threshold = 30
    upper_threshold = 70

    BB_n = 30
    STDDEV_n = 14

    ADX_n = 14
    ADX_lower = 22
    ADX_upper = 23

    ROC_n = 14
    MOM_n = 14

    def init(self):
        self.high = self.data.High
        self.low = self.data.Low
        self.close = self.data.Close

        self.sma = self.I(SMA, self.close, self.SMA_n)
        self.ema = self.I(EMA, self.close, self.EMA_n)
        self.rsi = self.I(RSI, self.close, self.RSI_n)

        self.macd, self.signal_macd, _ = talib.MACD(self.close, fastperiod=self.short_MA,
                                                    slowperiod=self.long_MA, signalperiod=self.signal_MA)

        self.slowk, self.slowd = talib.STOCH(self.high, self.low, self.close, fastk_period=self.fastk_period,
                                             slowk_period=self.slowk_period, slowd_period=self.slowd_period)

        self.bb_upper, _, self.bb_lower = talib.BBANDS(self.close, timeperiod=self.BB_n, nbdevup=2, nbdevdn=2)
        self.stddev = talib.STDDEV(self.close, timeperiod=self.STDDEV_n)
        self.adx = talib.ADX(self.high, self.low, self.close, timeperiod=self.ADX_n)
        self.roc = talib.ROCR(self.close, timeperiod=self.ROC_n)
        self.mom = talib.MOM(self.close, timeperiod=self.MOM_n)

    def next(self):
        if (not self.position and
                self.close[-1] > self.sma[-1] and
                self.close[-1] > self.ema[-1] and
                self.rsi[-1] < self.RSI_lower and
                self.macd[-1] > self.signal_macd[-1] and
                # self.slowk[-1] < self.lower_threshold
                # self.close[-1] < self.bb_lower[-1]
                self.roc[-1] > 0 and
                self.mom[-1] > 0
        ):
            self.buy()
            
        elif (self.close[-1] < self.sma[-1] and
              self.close[-1] < self.ema[-1] and
              self.rsi[-1] > self.RSI_upper and
              self.macd[-1] < self.signal_macd[-1] and
                # self.slowk[-1] > self.upper_threshold
                # self.close[-1] > self.bb_upper[-1]
                self.roc[-1] < 0 and
                self.mom[-1] < 0
        ):
            self.position.close()

bt = Backtest(df,Parameters,cash = 100000)
print(bt.run())

### optimaization method ==> (already run)
# opt_stats= bt.optimize(
#     SMA_n=range(3, 5, 1),
#     EMA_n=range(3, 5, 1),
#     RSI_n = range(7, 14, 1),
#     short_MA = range(5,7,1),
#     long_MA = range(15,18,1),
#     signal_MA = range(5,7,1),
#     fastk_period = range(5,14,1),
#     slowk_period = range(1,5,2),
#     ROC_n = range(7, 14, 1),
#     MOM_n = range(7, 14, 1)
#     )
# print(opt_stats._strategy)

'''
The best result is :
Parameters(SMA_n=3,EMA_n=3,RSI_n=7,short_MA=5,long_MA=15,signal_MA=5,fastk_period=5,slowk_period=1,ROC_n=7,MOM_n=7)
'''

### technical indicators

def cal_SMA(data, n):
    data[f"SMA_{n}"] = talib.SMA(data["Close"], n)
    return data

def cal_EMA(data, n):
    data[f"EMA_{n}"] = talib.EMA(data["Close"], n)
    return data

def cal_RSI(data, n=14):
    data["RSI"] = talib.RSI(data["Close"], timeperiod=n)
    return data

def cal_MACD(data, short_MA=12, long_MA=26, signal_MA=9):
    macd, signal, _ = talib.MACD(data["Close"], fastperiod=short_MA, slowperiod=long_MA, signalperiod=signal_MA)
    data["MACD_line"] = macd
    data["MACD_signal"] = signal
    return data

def cal_Stochastic_Oscillator(data, n=14, d=3):
    slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=n, slowk_period=d, slowd_period=d)
    data["slow_K"] = slowk
    data["slow_D"] = slowd
    return data

def cal_BBand(data, n):
    upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=n)
    data["BB_MA"] = middle
    data["BB_upper"] = upper
    data["BB_lower"] = lower
    return data

def cal_STDDEV(data, n):
    stddev = talib.STDDEV(data["Close"], timeperiod=n)
    data["STDDEV"] = stddev
    return data

def cal_ADX(data, n):
    data["ADX"] = talib.ADX(data["High"], data["Low"], data["Close"], timeperiod=n)
    return data

def cal_ROC(data, n=14):
    data["ROC"] = talib.ROCR(data["Close"], timeperiod=n)
    return data

def cal_MOM(data, n=14):
    data["MOM"] = talib.MOM(data["Close"], timeperiod=n)
    return data

df = cal_SMA(df,3)
df = cal_EMA(df,3)
df = cal_RSI(df,7)
df = cal_MACD(df,short_MA=5,long_MA=15,signal_MA=5)
df = cal_Stochastic_Oscillator(df,n=5, d=1)
df = cal_BBand(df,9)
df = cal_STDDEV(df,14)
df = cal_ADX(df,14)
df = cal_ROC(df,7)
df = cal_MOM(df,7)


df.dropna(inplace = True)
peaks,_ = find_peaks((-1)*df["Close"])
troughs,_ = find_peaks(-1*df["Close"])
df["positions"] = 0
df["positions"][peaks] = 1
df["positions"].value_counts()
data = df.drop(columns = ["Open","High","Low",'Adj Close'])
data.to_csv("feature_parameter_tuning.csv")

# plt.figure(figsize=(12, 6))
# plt.plot(data.index,data["Close"], label='Close Prices',color = "grey")
# plt.plot(data.index[peaks], data["Close"][peaks], 'g*', label='Peaks')
# plt.title('Peaks in USD/JPY Close Prices')
# plt.xlabel('Time')
# plt.ylabel('Close Price')
# plt.legend()
# plt.grid()
# plt.show()

# print(df)


