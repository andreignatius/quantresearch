import json
import numpy as np
import pandas as pd

####################################
## EXTRACT THE PNL FROM THE JSON FILES
####################################

# Read the first JSON file
file_path_1 = 'logreg_pnl_v_leverage.json'
with open(file_path_1, 'r') as file_1:
    data_1 = pd.read_json(file_1)

# Read the second JSON file
file_path_2 = 'gbt_pnl_v_leverage.json'
with open(file_path_2, 'r') as file_2:
    data_2 = pd.read_json(file_2)

# Read the third JSON file
file_path_3 = 'nn_pnl_v_leverage.json'
with open(file_path_3, 'r') as file_3:
    data_3 = json.load(file_3)

# Create DataFrames from each file's data
df_1 = pd.DataFrame(data_1.copy())  # Explicitly create a copy
df_2 = pd.DataFrame(data_2.copy())  # Explicitly create a copy
df_3 = pd.DataFrame(data_3.copy())  # Explicitly create a copy

# Find columns containing 'Leverage4' in each DataFrame
leverage4_columns_df1 = [col for col in df_1.columns if 'Leverage4' in col]
leverage4_columns_df2 = [col for col in df_2.columns if 'Leverage4' in col]
leverage4_columns_df3 = [col for col in df_3.columns if 'Leverage4' in col]

# Filter columns containing 'Leverage4' from both DataFrames
filtered_df_1 = df_1[leverage4_columns_df1].copy()  # Create a copy
filtered_df_2 = df_2[leverage4_columns_df2].copy()  # Create a copy
filtered_df_3 = df_3[leverage4_columns_df3].copy()  # Create a copy

# Rename columns
filtered_df_1.rename(columns={filtered_df_1.columns[0]: 'logreg_lev4'}, inplace=True)
filtered_df_2.rename(columns={filtered_df_2.columns[0]: 'gbt_lev4'}, inplace=True)
filtered_df_3.rename(columns={filtered_df_3.columns[0]: 'nn_lev4'}, inplace=True)

# Combine the filtered columns from both DataFrames
combined_filtered_df = pd.concat([filtered_df_1, filtered_df_2, filtered_df_3], axis=1)

# Keep the first row only
first_row = combined_filtered_df.iloc[[0]]

# Transpose the DataFrame
combined_filtered_df = first_row.T

####################################
## GET THE STD DEV OF RETURNS
####################################

# Load the JSON data from the first file (logreg_pnl_v_leverage.json)
with open('logreg_pnl_v_leverage.json', 'r') as file:
    data_logreg = json.load(file)
# Extract the PnL Distribution for Leverage4 from the first JSON file
leverage_4_pnl_distribution_logreg = data_logreg['Leverage4']['PnL Distribution']
# Calculate the standard deviation for Leverage4 from the first JSON file
std_dev_logreg = np.std(leverage_4_pnl_distribution_logreg)

# Load the JSON data from the second file (gbt_pnl_v_leverage.json)
with open('gbt_pnl_v_leverage.json', 'r') as file:
    data_gbt = json.load(file)
# Extract the PnL Distribution for Leverage4 from the second JSON file
leverage_4_pnl_distribution_gbt = data_gbt['Leverage4']['PnL Distribution']
# Calculate the standard deviation for Leverage4 from the second JSON file
std_dev_gbt = np.std(leverage_4_pnl_distribution_gbt)

# Load the JSON data from the second file (nn_pnl_v_leverage.json)
with open('nn_pnl_v_leverage.json', 'r') as file:
    data_nn = json.load(file)
# Extract the PnL Distribution for Leverage4 from the second JSON file
leverage_4_pnl_distribution_nn = data_nn['Leverage4']['PnL Distribution']
# Calculate the standard deviation for Leverage4 from the second JSON file
std_dev_nn = np.std(leverage_4_pnl_distribution_nn)

# Create a DataFrame to save the output
data = {
    'JSON File': ['logreg_pnl_v_leverage.json','gbt_pnl_v_leverage.json','nn_pnl_v_leverage.json'],
    'Standard Deviation': [std_dev_logreg, std_dev_gbt, std_dev_nn]
}

df = pd.DataFrame(data)
standard_deviation = df.iloc[:, 1]
standard_deviation = pd.DataFrame(standard_deviation, columns=['Standard Deviation'])

# # Reset the indices of both DataFrames
combined_filtered_df.reset_index(drop=True, inplace=True)
standard_deviation.reset_index(drop=True, inplace=True)

# # Concatenate the existing DataFrame with the new DataFrame containing the second column
combined_filtered_df = pd.concat([combined_filtered_df, standard_deviation], axis=1)

# # Create a list of values for the new column
new_column_values = ['logreg_lev4', 'gbt_lev4','nn_lev4']

# # Insert the new column at the first position (index 0)
combined_filtered_df.insert(0,'Model', new_column_values)

####################################
## GET THE MARKET RETURNS
####################################

# Load the Forex data
file_path = 'data/USD_JPY_YF.csv'
forex_data = pd.read_csv(file_path)

# Generate Market Return
investment = 10000  # Initial investment in USD

# Filter the forex_data DataFrame for 'Date' starting with '2013'
first_close_2013 = forex_data[forex_data['Date'].str.startswith('2013')]['Close'].iloc[0]

# Calculate the market return using the 'Close' values
starting_price = first_close_2013
final_price = forex_data['Close'].iloc[-1]

# Calculate the gains in the exchange rate
gains = (starting_price-final_price)/final_price

# Generate Output
market_returns = investment*(gains)

####################################
## CALCULATIONS AND SHARPE RATIO
####################################
combined_filtered_df['market_returns'] = market_returns
combined_filtered_df['excess_returns'] = combined_filtered_df['PnL']-market_returns
num_values_pnl = len(data_1['Leverage3']['PnL Distribution'])
combined_filtered_df['mean_returns'] = combined_filtered_df['excess_returns']/num_values_pnl
combined_filtered_df['sharpe_ratio'] = combined_filtered_df['mean_returns']/combined_filtered_df['Standard Deviation']

####################################
## CALCULATE EXCESS RETURNS
####################################

# Display the transposed DataFrame
print(combined_filtered_df)

# Save the transposed DataFrame as a CSV file
combined_filtered_df.to_csv('sharpe_ratiov2.csv')