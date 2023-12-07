# import subprocess
# import pandas as pd
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# # Run the script and capture its output
# process = subprocess.Popen(['python', 'logreg_oop.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# output, _ = process.communicate()

# # Decode the output and split it into lines
# output_lines = output.decode('utf-8').split('\n')

# # Filter the line containing "final_pnl"
# final_pnl_line = [line for line in output_lines if "final_pnl" in line]

# # Create a DataFrame with the filtered line
# df = pd.DataFrame({'Final_PNL_Line': final_pnl_line})

# # Display the DataFrame
# print(df)

##########################

import subprocess
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

leverage_factors = [x / 2.0 for x in range(2, 11)]  # Generate leverage_factor values from 1 to 5 in increments of 0.5

final_pnl_lines = []

for leverage_factor in leverage_factors:
    # Run the script and capture its output
    process = subprocess.Popen(['python', 'logreg_oop.py'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               stdin=subprocess.PIPE)
    
    # Generate input string to pass leverage_factor to the script
    input_string = f"{leverage_factor}\n"
    output, _ = process.communicate(input=input_string.encode())
    
    # Decode the output and split it into lines
    output_lines = output.decode('utf-8').split('\n')
    
    # Filter the line containing "final_pnl"
    final_pnl_line = [line for line in output_lines if "final_pnl" in line]
    
    # Append the final_pnl_line to the list
    final_pnl_lines.extend(final_pnl_line)

# Create a DataFrame with the filtered lines
df = pd.DataFrame({'Final_PNL_Line': final_pnl_lines})

# Display the DataFrame
print(df)
