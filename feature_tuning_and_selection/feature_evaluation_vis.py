import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('feature_evaluation.csv')

# Group data by 'Source CSV'
grouped = data.groupby('Source CSV')

# Create a directory to save the visualizations
output_directory = 'visualizations'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Create tables and heatmaps for each group and save them in the directory
for name, group in grouped:
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = group.pivot_table(index='Technical Indicator', columns='Source CSV', values=['Importance to Peak', 'Importance to Trough'])
    
    sns.heatmap(pivot['Importance to Peak'], annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title(f'Importance to Peak - {name}')
    plt.xlabel('Source CSV')
    plt.ylabel('Technical Indicator')
    plt.tight_layout()

    # Save the table with heatmap in the 'visualizations' folder
    plt.savefig(os.path.join(output_directory, f'{name}_Importance_to_Peak.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot['Importance to Trough'], annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title(f'Importance to Trough - {name}')
    plt.xlabel('Source CSV')
    plt.ylabel('Technical Indicator')
    plt.tight_layout()

    # Save the table with heatmap in the 'visualizations' folder
    plt.savefig(os.path.join(output_directory, f'{name}_Importance_to_Trough.png'))
    plt.close()
