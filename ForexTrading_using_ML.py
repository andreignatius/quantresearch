"""
Input feature: technical indicators
Period: 200 days
Target label: peaks and troughs
predict the trend of the SGD/CNY
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import yfinance as yf
import talib
from scipy.signal import find_peaks

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

y_ticker = "SGDCNY=X"
end = dt.datetime.now()
start = end - dt.timedelta(days = 300)
LAG = 5
MAs = [5,14,21,63]
EX_MAs = [3,5,7,21]
RSIs = [3,5,14,21]
K_Ds = [5,14,21]
D_smooth = 3
ROCs =[5,14,21]
MOMs = [5,14,21]

#### Get input and output features
def cal_LAG(data,LAG):
    for lag in range(1,LAG):
        data[f"lag_{lag}"] = np.log(data["Close"]).diff(lag)
    return data

def cal_SMA(data,n):
    data[f"SMA_{n}"] = talib.SMA(data["Close"],n)
    return data

def cal_EMA(data,n):
    data[f"EMA_{n}"] = talib.EMA(data["Close"],n)
    return data

def cal_RSI(data,n=14):
    data[f"RSI_{n}"] = talib.RSI(data["Close"], timeperiod=n)
    return data

def cal_Stochastic(data,n=14, d=D_smooth):
    slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'],fastk_period=n, slowk_period=d, slowd_period=d)
    data[f"slow_K_{n}"] = slowk
    data[f"slow_D_{n}"] = slowd
    return data

def cal_ROC(data,n=14):
    data[f"ROC_{n}"] = talib.ROCR(data["Close"], timeperiod=n)
    return data
def cal_MOM(data,n=14):
    data[f"MOM_{n}"] = talib.MOM(data["Close"],timeperiod = n)
    return data

def get_data(y_ticker,LAG, MAs, EX_MAs, RSIs, K_Ds, ROCs,MOMs):
    outcome = yf.download(y_ticker, start=start, end=end)

    # technical indicators
    outcome = cal_LAG(outcome,LAG)
    for i in MAs:
        outcome = cal_SMA(outcome, i)
    for j in EX_MAs:
        outcome = cal_EMA(outcome, j)
    for p in RSIs:
        outcome = cal_RSI(outcome, p)
    for q in K_Ds:
        outcome = cal_Stochastic(outcome, q)
    for k in ROCs:
        outcome = cal_ROC(outcome, k)
    for m in MOMs:
        outcome = cal_MOM(outcome, m)
    outcome = outcome.drop(columns = ["Open","High","Low","Adj Close","Volume"])
    outcome.dropna(inplace = True)
    close_prices = outcome['Close'].to_numpy()
    through,_ = find_peaks((-1) * close_prices)
    outcome["y"] = 0
    outcome["y"][through +1] = 1 # Why need +1 here

    vis_signals(outcome)
    vis_Boxplot(outcome)
    vis_Histogram(outcome)

    return outcome

def vis_signals(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index,data.Close, label='Close Prices')
    plt.plot(data[data["y"].shift(-1)==1].index,
             data[data["y"].shift(-1)==1].Close, 'r*', label='Troughs')
    # plt.plot(data[data["y"]==-1].index, data[data["y"]==-1].Close, 'g*', label='Troughs')
    plt.title('Peaks and Troughs in SGD/CNY Close Prices')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()

def vis_Boxplot(data):
    num = data.shape[1]
    fig, axs = plt.subplots(ncols=5, nrows=int(num/5)+1, figsize=(18, 15))
    index = 0
    axs = axs.flatten()
    for k, v in data.items():
        sns.boxplot(y=k, data=data, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()

def vis_Histogram(data):
    num = data.shape[1]
    fig, axs = plt.subplots(ncols=5, nrows=int(num/5)+1, figsize=(18, 15))
    index = 0
    axs = axs.flatten()
    for k, v in data.items():
        sns.histplot(v, ax=axs[index], bins=50)
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()

df = get_data(y_ticker,LAG, MAs, EX_MAs, RSIs, K_Ds, ROCs,MOMs)


validation_size = 0.25
train_size = int(len(df) * (1 - validation_size))
df_train = df[:train_size]
df_test = df[train_size:len(df)]

def Algorithm_comparison(df_train,df_test,seed = 1212):
    x_train = df_train.drop(columns = ["Close","y"])
    y_train = df_train["y"]
    x_test = df_test.drop(columns=["Close", "y"])
    y_test = df_test["y"]

    x_train = pd.DataFrame(StandardScaler().fit_transform(x_train),index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(StandardScaler().fit_transform(x_test), index=x_test.index, columns=x_test.columns)

    models = []
    models.append(("Logit", LogisticRegression(random_state=seed)))
    models.append(("Lasso", Lasso(random_state=seed)))
    models.append(("Ridge", Ridge(random_state=seed)))
    models.append(("EN", ElasticNet(random_state=seed)))
    models.append(("SVR", SVC()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("CART", DecisionTreeClassifier(random_state=seed)))
    models.append(("ETR", ExtraTreesClassifier(random_state=seed)))
    models.append(("RFR", RandomForestClassifier(random_state=seed)))
    models.append(("GBR", GradientBoostingClassifier(random_state=seed)))
    models.append(("ABR", AdaBoostClassifier(random_state=seed)))

    names = []
    train_MSEs = []
    train_RMSEs = []
    train_R2s = []
    test_MSEs = []
    test_R2s = []
    test_RMSEs = []

    for name, model in models:
        names.append(name)
        res = model.fit(x_train, y_train)

        y_pred_train = res.predict(x_train)
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_MSEs.append(train_mse)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_RMSEs.append(train_rmse)
        train_r2 = r2_score(y_train, y_pred_train)
        train_R2s.append(train_r2)

        y_pred_test = res.predict(x_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_MSEs.append(test_mse)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_RMSEs.append(test_rmse)
        test_r2 = r2_score(y_test, y_pred_test)
        test_R2s.append(test_r2)

    df_train_perform = pd.DataFrame({"MSE": train_MSEs,"RMSE": train_RMSEs,
                                     "R_square": train_R2s}, index=names)
    df_test_perfrom = pd.DataFrame({"MSE": train_MSEs,"RMSE": test_RMSEs,
                                    "R_square": test_R2s}, index=names)
    df_perform = pd.concat([df_train_perform, df_test_perfrom], axis=1, keys=["train", "test"])
    vis_train_test_errors(names, train_MSEs, test_MSEs)
    return df_perform,x_train, x_test,y_train, y_test,models

def vis_train_test_errors(names,train_results,test_results):
    fig = plt.figure(figsize = [10,6])
    ind = np.arange(len(names))
    width = 0.30
    fig.suptitle("Comparing the Perfomance of Various Algorithms on the Training vs. Testing Data")
    ax = fig.add_subplot(111)
    plt.bar(ind - width/2,train_results,width = width,label = "Errors in Training Set")
    plt.bar(ind + width/2,test_results,width = width,label = "Errors in Testing Set")
    plt.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    plt.ylabel("Mean Squared Error (MSE)")
    plt.show()

"""
Considering the performances of different algorithms and comparing with the MSE, RMSE, and R_square, we choose Lasso regression as our best model to do prediction.
"""

def vis_performance(best_model,models,x_train, x_test,
                    y_train, y_test):
    for name, model_i in models:
        if name == best_model:
            target_model = model_i
        else:
            continue
    model = target_model
    print(target_model)
    prediction = model.fit(x_train,y_train).predict(x_test)
    score = model.fit(x_train,y_train).score(x_test,y_test)
    print(f"Algorithm:          {best_model}")
    print(f"The score:          {round(score,20)}")

    y_predicted = pd.DataFrame(model.predict(x_test),index = y_test.index)
    y_test.to_frame()
    plt.figure(figsize = (12,8))
    plt.plot(y_test.index,y_test,"b--",lw = 0.3, label = "Actual Y")
    plt.plot(y_predicted.index,y_predicted,"r",lw = 0.3, label = "Predicted Y (Y hat)")
    plt.ylim([-2,2])
    plt.legend()
    plt.show()
    return prediction


PERFROMS,X_train, X_test,Y_train, Y_test,models = Algorithm_comparison(df_train,df_test)
print(PERFROMS)

predictions = vis_performance("Ridge",models,X_train, X_test,Y_train, Y_test)
df_test["position"] = np.where(predictions > 0,1,0)
capital = 100000
commission = 5

def calculate_strategy_return(signals, capital, commission):
    portfolio = pd.DataFrame(index=signals.index)
    portfolio["price"] = signals["Close"]
    portfolio["position"] = signals["position"]
    portfolio["signal"] = portfolio["position"].diff()
    portfolio["returns"] = portfolio["price"].pct_change()
    portfolio["adj_returns"] = portfolio["position"] * portfolio["returns"]
    portfolio.dropna(inplace=True)
    portfolio["FX"] = (1 + portfolio["returns"]).cumprod()
    portfolio["total"] = np.nan
    temp_value = capital
    for idx in portfolio.index:
        temp_value = (np.exp(portfolio.loc[idx]["adj_returns"]) *
                      (temp_value - commission * abs(portfolio.loc[idx]["signal"])))
        portfolio.loc[idx, "total"] = temp_value

    portfolio["str_ret"] = portfolio["total"].pct_change()
    portfolio.dropna(inplace=True)
    portfolio["cum_ret"] = (1 + portfolio["returns"]).cumprod()
    portfolio["cum_str_ret"] = (1 + portfolio["str_ret"]).cumprod()
    portfolio["FX_max_perform"] = portfolio["cum_ret"].cummax()
    portfolio["str_max_perform"] = portfolio["cum_str_ret"].cummax()
    # portfolio.dropna(inplace = True)

    # FX
    FX_final_account = portfolio["FX"][-1] * capital
    FX_cumulative_returns = portfolio["cum_ret"][-1]

    FX_average_return = portfolio["returns"].mean()
    FX_std_return = portfolio["returns"].std()
    FX_Sharpe_ratio = np.sqrt(253) * (FX_average_return / FX_std_return)

    FX_days = (portfolio.index[-1] - portfolio.index[0]).days
    FX_CAGR = (portfolio["total"][-1] / portfolio["total"][0]) ** (365 / FX_days) - 1

    FX_drawdown = portfolio["FX_max_perform"] - portfolio["cum_ret"]
    FX_max_drawdown = FX_drawdown.max()
    FX_drawdown_period = (FX_drawdown[FX_drawdown == 0].index[1:] - FX_drawdown[FX_drawdown == 0].index[:-1]).days
    FX_longest_drawdown_period = FX_drawdown_period.max()

    # Strategy
    strategy_final_account = portfolio["total"][-1]
    cumulative_returns = portfolio["cum_str_ret"][-1]

    average_daily_return = portfolio["str_ret"].mean()
    daily_std = portfolio["str_ret"].std()
    Sharpe_ratio = np.sqrt(253) * (average_daily_return / daily_std)

    days = (portfolio.index[-1] - portfolio.index[0]).days
    CAGR = (portfolio["total"][-1] / portfolio["total"][0]) ** (365 / days) - 1

    drawdown = portfolio["str_max_perform"] - portfolio["cum_str_ret"]
    max_drawdown = drawdown.max()
    drawdown_period = (drawdown[drawdown == 0].index[1:] - drawdown[drawdown == 0].index[:-1]).days
    longest_drawdown_period = drawdown_period.max()

    plt.figure(figsize=(16, 10))
    plt.plot(portfolio.index, portfolio["cum_ret"], label="SPY CumulativeReturns", color="green")
    plt.plot(portfolio.index, portfolio["cum_str_ret"], label="Strategy CumulativeReturns", color="red")
    plt.legend()
    plt.show()

    print(f"The performance of FX is:")
    print(f"    Sharpe ratio:            {FX_Sharpe_ratio}")
    print(f"    CAGR:                    {FX_CAGR}")
    print(f"    Max drawdown:            {FX_max_drawdown}")
    print(f"    Longest drawdown period: {FX_longest_drawdown_period} days")
    print(f"    Finanl account:          {FX_final_account}")

    print("================================================================")
    print("================================================================")

    print(f"The performance of Strategy is:")

    print(f"    Sharpe ratio:            {Sharpe_ratio}")
    print(f"    CAGR:                    {CAGR}")
    print(f"    Max drawdown:            {max_drawdown}")
    print(f"    Longest drawdown period: {longest_drawdown_period} days")
    print(f"    Finanl account:          {strategy_final_account}")
    return portfolio


STRATEGY = calculate_strategy_return(df_test, capital, commission)
print(STRATEGY.iloc[:,3:8])
