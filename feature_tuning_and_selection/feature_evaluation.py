import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Pull data from feature parameter tuning and technical indicators
df_1 = pd.read_csv('feature_parameter_tuning.csv')
# print(len(df_1))
df_2 = pd.read_csv('technical_indicators.csv')
# print(len(df_2))
df_peak = pd.read_csv('peak_detection.csv')
# print(len(df_peak))

# Perform left join on 'date' column
df_3 = pd.merge(df_2, df_1, on='Date', how='left')

# Perform left join on 'date' column
df_3 = pd.merge(df_3, df_peak, on='Date', how='left')

# Bring in Hurst, Kalman, FFT, and higher order derivatives
df_4 = pd.read_csv('hurst_kalman_fft_derivatives.csv')
df_3 = pd.merge(df_3, df_4, on='Date', how='left')

# Drop columns
columns_to_drop = ['Unnamed: 0', 'Date', 'Open', 'High', 'Low',\
                   'Close_x', 'Close_y', 'Adj Close','Volume',\
                    'EoM','Trend_Up','positions']
df_3.drop(columns=columns_to_drop, inplace=True)

# Drop rows
df_3.dropna(inplace=True)

# Print dataframe
pd.set_option('display.max_columns', None)
# print(df_3.tail())

#### Cleanup: remove identical columns by checking their data
# Finding columns with identical values
duplicate_cols = set()
for i in range(len(df_3.columns)):
    col_i = df_3.iloc[:, i]
    for j in range(i + 1, len(df_3.columns)):
        col_j = df_3.iloc[:, j]
        if col_i.equals(col_j):
            duplicate_cols.add(df_3.columns[j])

# Keeping the first instance and dropping the duplicates
df_3 = df_3.drop(columns=duplicate_cols)


#### remove identical columns based on name
# List of indicators that appear twice
duplicated_indicators = ['ADX', 'Long_Moving_Avg', 'RSI_y', 'MACD_signal']

# Assuming df_3 is your DataFrame containing these indicators

# Drop one instance of each duplicated indicator
for indicator in duplicated_indicators:
    if indicator in df_3.columns:
        # Find the indices of the duplicated columns
        duplicate_indices = df_3.columns.get_loc(indicator)
        if isinstance(duplicate_indices, int):  # Only one instance exists
            continue
        else:
            # Drop one instance (keeping the first one)
            df_3.drop(df_3.columns[duplicate_indices[0]], axis=1, inplace=True)

# Now df_3 will have one instance of each indicator that appeared twice
print(df_3.columns)  # Printing the remaining columns in df_3

# Generate list of columns
columns_list = df_3.columns.tolist()
print(columns_list)

############################################
## DATA CLEANUP COMPLETED, FEATURE ANALYSIS
############################################

# Separate features and target variables
X = df_3.drop(['Peak', 'Trough'], axis=1)  # Features
y = df_3[['Peak', 'Trough']]  # Target variables

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier for each target variable (Random Forest Feature Importance)
# other possible methodologies - recursive feature elimination (RFE), Gradient Boosting, Lasso Regression
clf_peak = RandomForestClassifier(random_state=42)
clf_trough = RandomForestClassifier(random_state=42)

clf_peak.fit(X_train, y_train['Peak'])
clf_trough.fit(X_train, y_train['Trough'])

# Feature Importance for Peak
feature_importance_peak = clf_peak.feature_importances_
feature_names = X.columns

# Create a DataFrame to store feature importance for Peak
feature_importance_peak_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_peak})
feature_importance_peak_df = feature_importance_peak_df.sort_values(by='Importance', ascending=False)

# Plotting Feature Importance for Peak
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance_peak_df, palette='viridis')
ax.tick_params(axis='y', labelsize=5)  # Set the desired font size here
plt.title('Feature Importance for Peak')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Feature Importance for Trough
feature_importance_trough = clf_trough.feature_importances_

# Create a DataFrame to store feature importance for Trough
feature_importance_trough_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_trough})
feature_importance_trough_df = feature_importance_trough_df.sort_values(by='Importance', ascending=False)

# Plotting Feature Importance for Trough
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Importance', y='Feature', data=feature_importance_trough_df, palette='viridis')
ax.tick_params(axis='y', labelsize=5)  # Set the desired font size here
plt.title('Feature Importance for Trough')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Correlation Analysis (DEPRECATED)
# correlation_matrix = df_3.corr()

# Plotting Correlation Matrix
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Matrix')
# plt.show()

# print(df_3.shape)