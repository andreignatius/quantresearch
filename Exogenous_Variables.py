import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import spearmanr

# Load data
file_path_JPYTBill = 'backtest/data/JPY_1Y_TBill_GJTB12MO.csv'
file_path_USTBill = 'backtest/data/USD_1Y_TBill_H15T1Y.csv'
file_path_JPYCPI = 'backtest/data/japan_CPI.xlsx'
file_path_USCPI = 'backtest/data/US_CPI.xlsx'
file_path = 'backtest/data/final_dataset_with_new_features.csv'
file_path_CurrencyAccount_JP = 'backtest/data/Japan_Currency_Account.xlsx'
file_path_CurrencyAccount_US = 'backtest/data/US_Currency_Account.xlsx'
file_path_FDI_JP = 'backtest/data/Japan_FDI.xlsx'
file_path_FDI_US = 'backtest/data/USA_FDI.xlsx'
file_path_M2_JP = 'backtest/data/Japan_Monetary_Supply.xlsx'
file_path_M2_US = 'backtest/data/US_Monetary_Supply.xlsx'

TBill_data_JP = pd.read_csv(file_path_JPYTBill)
TBill_data_US = pd.read_csv(file_path_USTBill)
CPI_data_JP = pd.read_excel(file_path_JPYCPI, parse_dates = True)
CPI_data_US = pd.read_excel(file_path_USCPI, parse_dates = True)
CurrencyAccount_JP = pd.read_excel(file_path_CurrencyAccount_JP, parse_dates= True)
CurrencyAccount_US = pd.read_excel(file_path_CurrencyAccount_US, parse_dates= True)
FDI_JP = pd.read_excel(file_path_FDI_JP, parse_dates= True)
FDI_US = pd.read_excel(file_path_FDI_US, parse_dates= True)
M2_JP = pd.read_excel(file_path_M2_JP, parse_dates= True)
M2_US = pd.read_excel(file_path_M2_US, parse_dates= True)
forex_data = pd.read_csv(file_path, parse_dates= True)

scaler = StandardScaler()

# get TBill data
TBill_data = TBill_data_US.merge(TBill_data_JP, on = 'Date')
TBill_data['Interest_Rate_Difference'] = TBill_data['PX_LAST_y'] - TBill_data['PX_LAST_x']
TBill_data['Interest_Rate_Difference_Change'] = TBill_data['Interest_Rate_Difference'].diff()
TBill_data = TBill_data[['Date','Interest_Rate_Difference_Change']]
TBill_data['Date'] = pd.to_datetime(TBill_data['Date'], dayfirst= True)

# get CPI data
CPI_data = CPI_data_US.merge(CPI_data_JP, on = 'Date')
CPI_data['CPI_Difference'] = CPI_data['PX_LAST_y'] / CPI_data['PX_LAST_x']
CPI_data = CPI_data[['Date','CPI_Difference']]
CPI_data = CPI_data.set_index('Date')
# Linear Interpolate the data
CPI_data = CPI_data.resample('D').interpolate()
CPI_data['CPI_Difference'] = CPI_data['CPI_Difference'].interpolate()
CPI_data['CPI_Difference_Change'] = CPI_data['CPI_Difference'].diff()
CPI_data = CPI_data.drop(columns = 'CPI_Difference')
CPI_data = CPI_data.reset_index()

# get Currency Account data
CurrencyAccount_data = CurrencyAccount_US.merge(CurrencyAccount_JP, on ='Date')
CurrencyAccount_Date = CurrencyAccount_data['Date'].values
CurrencyAccount_data = CurrencyAccount_data.set_index('Date')
CurrencyAccount_data = CurrencyAccount_data[['PX_LAST_x', 'PX_LAST_y']]
CurrencyAccount_data_scaled = scaler.fit_transform(CurrencyAccount_data)
CurrencyAccount_data = pd.DataFrame(CurrencyAccount_data_scaled, index= CurrencyAccount_Date)
CurrencyAccount_data['Currency_Account_difference'] = CurrencyAccount_data.iloc[:,1] - CurrencyAccount_data.iloc[:,0]
CurrencyAccount_data = CurrencyAccount_data[['Currency_Account_difference']]
# Linear Interpolate the data
CurrencyAccount_data = CurrencyAccount_data.resample('D').interpolate()
CurrencyAccount_data['Currency_Account_difference'] = CurrencyAccount_data['Currency_Account_difference'].interpolate()
CurrencyAccount_data = CurrencyAccount_data.reset_index()
CurrencyAccount_data.columns = ['Date', 'Currency_Account_difference']

# get FDI data
FDI_data = FDI_US.merge(FDI_JP, on= 'Date')
FDI_data['FDI_Difference'] = FDI_data['PX_LAST_y'] - FDI_data['PX_LAST_x']
FDI_data = FDI_data[['Date', 'FDI_Difference']]
FDI_data = FDI_data.set_index('Date')
# Linear Interpolation
FDI_data = FDI_data.resample('D').interpolate()
FDI_data['FDI_Difference'] = FDI_data['FDI_Difference'].interpolate()
FDI_data['FDI_Difference_Change'] = FDI_data['FDI_Difference'].diff()
FDI_data = FDI_data.drop(columns = 'FDI_Difference')
FDI_data = FDI_data.reset_index()

# get M2
M2_data = M2_US.merge(M2_JP, on = 'Date')
M2_data['M2_Difference'] = M2_data['PX_LAST_y']/ M2_data['PX_LAST_x']
M2_data = M2_data[['Date', 'M2_Difference']]
M2_data = M2_data.set_index('Date')
# Linear Interpolation
M2_data = M2_data.resample('D').interpolate()
M2_data['M2_Difference'] = M2_data['M2_Difference'].interpolate()
M2_data['M2_Difference_Change'] = M2_data['M2_Difference'].diff()
M2_data = M2_data.drop(columns = 'M2_Difference')
M2_data = M2_data.reset_index()


# merge the figures
forex_data['Date'] = pd.to_datetime(forex_data['Date'])
forex_data = forex_data.merge(CPI_data, on = 'Date')
forex_data = forex_data.merge(TBill_data, on = 'Date')
forex_data = forex_data.merge(CurrencyAccount_data, on = 'Date')
forex_data = forex_data.merge(FDI_data, on = 'Date')
forex_data = forex_data.merge(M2_data, on= 'Date')

# Check correlation
forex_data['Signal'] = np.where(forex_data['Label'] == 'Buy', 1, 0)
forex_data['Signal'] = np.where(forex_data['Label'] == 'Sell', -1, forex_data['Signal'])
Correlation_Checking = forex_data[['CPI_Difference_Change',
       'Interest_Rate_Difference_Change', 'Currency_Account_difference',
       'FDI_Difference_Change', 'M2_Difference_Change', 'Signal']]
# Normal Correlation
Correlation = Correlation_Checking.corr()
# Spearman Correlation
Correlation_coef_spearman, p_value = spearmanr(Correlation_Checking.iloc[:,:-1], Correlation_Checking['Signal'])

# Drop Currency Account Difference Feature as the correlation and Spearman Correlation with Signal is low
forex_data = forex_data.drop(columns= ['Currency_Account_difference', 'Signal'])

print(forex_data.columns)
print(Correlation, Correlation_coef_spearman)

