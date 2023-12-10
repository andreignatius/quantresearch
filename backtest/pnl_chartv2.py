import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to calculate cumulative PNL
def calculate_cumulative_pnl(pnl_distribution):
    return [sum(pnl_distribution[:i+1]) for i in range(len(pnl_distribution))]

# Load data from the first JSON file (logreg_pnl_v_leverage.json)
with open('logreg_pnl_v_leverage.json', 'r') as file1:
    logreg_data = json.load(file1)

# Load data from the second JSON file (gbt_pnl_v_leverage.json)
with open('gbt_pnl_v_leverage.json', 'r') as file2:
    gbt_data = json.load(file2)

# Extract PNL data for leverage 4 from both files
pnl_distribution_leverage_4_logreg = logreg_data.get('Leverage4', {}).get('PnL Distribution', [])
pnl_distribution_leverage_4_gbt = gbt_data.get('Leverage4', {}).get('PnL Distribution', [])

# Calculate cumulative PNL for both sets of data
cumulative_pnl_logreg = calculate_cumulative_pnl(pnl_distribution_leverage_4_logreg)
cumulative_pnl_gbt = calculate_cumulative_pnl(pnl_distribution_leverage_4_gbt)

# Generate x-axis labels for half-yearly periods starting from 2013
periods = ['2013-01-30', '2013-06-30', '2014-01-30', '2014-06-30',\
           '2015-01-30', '2015-06-30', '2016-01-30', '2016-06-30',\
           '2017-01-30', '2017-06-30', '2018-01-30', '2018-06-30',\
           '2019-01-30', '2019-06-30', '2029-01-30', '2020-06-30',\
           '2021-01-30', '2021-06-30', '2022-01-30']

# Creating a DataFrame with cumulative PNL data and periods
data = {
    'Period': periods[:len(cumulative_pnl_logreg)],  # Adjusting periods to match data length
    'Cumulative_PNL_LogReg': cumulative_pnl_logreg,
    'Cumulative_PNL_GBT': cumulative_pnl_gbt
}

# df = pd.DataFrame(data)

#####################################################

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download S&P 500 data from Yahoo Finance
sp500_data = yf.download('^GSPC', start='2013-01-01', end='2023-12-31')['Close']

# Download USD/JPY forex data from Yahoo Finance
usdjpy_data = yf.download('USDJPY=X', start='2013-01-01', end='2023-12-31')['Close']

# Align date indices and handle missing values
sp500_data, usdjpy_data = sp500_data.align(usdjpy_data, join='inner')

# Assuming an initial investment of $10,000
initial_investment = 10000

# Calculate PNL for investing in S&P 500
investment_value_sp500 = initial_investment * (sp500_data / sp500_data.iloc[0])
pnl_sp500 = investment_value_sp500 - initial_investment

# Calculate PNL for investing in USD/JPY (inverse exchange rate)
usd_to_usdjpy = 1 / usdjpy_data
investment_value_usdjpy = initial_investment * (usd_to_usdjpy / usd_to_usdjpy.iloc[0])
pnl_usdjpy = investment_value_usdjpy - initial_investment

# Combine the date index and PNL values into a DataFrame
pnl_data = pd.DataFrame({
    'Date': sp500_data.index,  # Assuming both sp500_data and usdjpy_data have the same index
    'USD_JPY PnL': pnl_usdjpy.values,
    'S&P500 PnL': pnl_sp500.values
})

# Display the DataFrame
# print(pnl_data)

#####################################################

# Convert 'Period' column to datetime format in the first DataFrame
data['Period'] = pd.to_datetime(data['Period'])

# Create a DataFrame from the 'data' dictionary
data_df = pd.DataFrame(data)

# Merge both DataFrames on the 'Date' column
combined_data = pd.merge(pnl_data, data_df, how='inner', left_on='Date', right_on='Period')

# Drop the extra 'Period' column (if needed)
combined_data = combined_data.drop(columns=['Period'])

# Display the combined DataFrame
print(combined_data)

#####################################################

# Plotting the data
plt.figure(figsize=(10, 6))  # Set the figure size

# Plot S&P 500 PnL
plt.plot(combined_data['Date'], combined_data['S&P500 PnL'], label='S&P 500 PnL')

# Plot USD/JPY PnL
plt.plot(combined_data['Date'], combined_data['USD_JPY PnL'], label='USD/JPY PnL')

# Plot Cumulative PNL for LogReg and GBT
plt.plot(combined_data['Date'], combined_data['Cumulative_PNL_LogReg'], label='Cumulative PNL LogReg')
plt.plot(combined_data['Date'], combined_data['Cumulative_PNL_GBT'], label='Cumulative PNL GBT')

plt.xlabel('Date')  # Set the label for the x-axis
plt.ylabel('Profit/Loss')  # Set the label for the y-axis
plt.title('PNL Performance from 2013 to 2021')  # Set the plot title
plt.legend()  # Show legend

plt.grid(True)  # Show gridlines
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Save the plot as a PNG file
plt.savefig('pnl_chartv2.png', dpi=300, bbox_inches='tight')  # Save the plot as PNG

plt.show()  # Display the plot