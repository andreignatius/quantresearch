import pandas as pd
import warnings
from logreg_oop import rolling_window_train_predict
import os
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Generate leverage_factor values from 1 to 3 in increments of 1
# leverage_factors = [x for x in range(1, 4)]

# Generate leverage_factor values from 1 to 5 in increments of 0.5
# leverage_factors = [x / 2 for x in range(2, 11)]

# Generate leverage factors
leverage_factors = [1, 1.5, 2, 3, 4]


# Load the Forex data
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, 'data/USD_JPY_YF.csv')
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
data.sort_values('Date', inplace=True)       # Sort by 'Date'

# Create an empty DataFrame to store leverage and P&L
pnl_data = pd.DataFrame(columns=['Leverage', 'PnL'])

# Cycle through leverage factors
for leverage_factor in leverage_factors:
    # Obtain P&L values
    trade_logs, final_values, _, _ = rolling_window_train_predict(data, 2013, 2023, 12, 6, leverage_factor=leverage_factor)  # 12 months training, 6 months testing
    pnl_per_quarter = [x - 10000 for x in final_values]
    total_pnl = sum(pnl_per_quarter)

    # Append leverage and P&L to the DataFrame using pd.concat
    new_row = pd.DataFrame({'Leverage': [leverage_factor], 'PnL': [total_pnl]})
    pnl_data = pd.concat([pnl_data, new_row], ignore_index=True)

# Display the DataFrame
print(pnl_data)

# Save DataFrame to a CSV file
pnl_data.to_csv('logreg_pnl_v_leverage.csv', index=False)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(pnl_data['Leverage'], pnl_data['PnL'], marker='o', linestyle='-')
plt.title('PnL vs Leverage')
plt.xlabel('Leverage')
plt.ylabel('PnL')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
plt.savefig('logreg_pnl_v_leverage.png')

























