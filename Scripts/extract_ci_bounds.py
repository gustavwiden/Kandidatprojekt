import os
import pandas as pd
from utils import base_dir

# --- Configuration ---
# Using your centralized base_dir to reliably target the Results folder
target_dir = os.path.join(base_dir, 'Results', 'Profile_likelihood')
output_txt_file = os.path.join(target_dir, "95_CI_Bounds_400_wide.txt")

# Dictionary of parameter names and their corresponding index in the merged array
parameters = {
    'F': 0,
    'ka': 1,
    'RC2': 13,
    'CL_HV': 15,
    'CL_SLE': 16,
    'kdeg': 17
}

# The naming convention used in your PL script
file_pattern = "acceptable_params_PL_{}_400.csv"

# --- Extraction Logic ---
print(f"Scanning directory: {target_dir}\n")

# Create output directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

with open(output_txt_file, 'w') as f_out:
    # Write a header for the text file
    f_out.write("95% Confidence Intervals from Profile Likelihood\n")
    f_out.write("================================================\n\n")
    
    for param_name, param_idx in parameters.items():
        csv_filename = file_pattern.format(param_name)
        csv_path = os.path.join(target_dir, csv_filename)
        
        if os.path.exists(csv_path):
            try:
                # Read the CSV. header=None ensures the first row isn't treated as column names
                df = pd.read_csv(csv_path, header=None)
                
                # Extract the min and max from the specific parameter's column
                lower_bound = df[param_idx].min()
                upper_bound = df[param_idx].max()
                
                # Format the output string
                result_line = f"{param_name}:\t[{lower_bound:.4f}, {upper_bound:.4f}]"
                print(f"Found {result_line}")
                f_out.write(result_line + "\n")
                
            except Exception as e:
                error_line = f"{param_name}:\tError processing file -> {e}"
                print(error_line)
                f_out.write(error_line + "\n")
        else:
            missing_line = f"{param_name}:\tCSV not found ({csv_filename})"
            print(missing_line)
            f_out.write(missing_line + "\n")

print(f"\nSuccess! All bounds have been saved to:\n{output_txt_file}")