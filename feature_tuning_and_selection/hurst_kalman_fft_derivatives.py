import pandas as pd

df = pd.read_csv('../exogenous_variables.csv',\
                 parse_dates=["Date"]).drop(columns = "Volume")

# List of columns to drop
columns_to_drop = ['Open', 'Close', 'High', 'Low', 'Adj Close', 'Unnamed: 0','Label', 'Signal']

# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True)

# Save the DataFrame to a CSV file
df.to_csv('hurst_kalman_fft_derivatives.csv', index=False)

# Show a list of the remaining columns
# columns_list = df.columns.tolist()
# print(columns_list)

# Set display options to show all columns without truncation
# pd.set_option('display.max_columns', None)
# print(df)

# print(df.shape)