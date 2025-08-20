# Import necessary libraries
import pandas as pd
import os

# Create save directory
save_dir = '../../Results/Acceptable params'
os.makedirs(save_dir, exist_ok=True)

# Define a function that calculates 95 % CI from MCMC
def calculate_quantiles_from_MCMC():
    in_path = os.path.join(save_dir, 'sampling_result.csv')
    # in_path = os.path.join(save_dir, 'sampling_result_SLE.csv')
    df = pd.read_csv(in_path, header=None)
    df.columns = ['F', 'ka', 'RC2', 'CL', 'kdegp']
    # df.columns = ['CL']

    # Store confidence intervals
    CI_data = []
    for col in df.columns:
        lower = df[col].quantile(0.025)
        upper = df[col].quantile(0.975)
        CI_data.append([col, lower, upper])

    # Save to CSV
    out_path = os.path.join(save_dir, "MCMC_param_confidence_intervals.csv")
    # out_path = os.path.join(save_dir, "MCMC_SLE_param_confidence_intervals.csv")
    pd.DataFrame(CI_data, columns=["Parameter", "Lower_95CI", "Upper_95CI"]).to_csv(out_path, index=False)

calculate_quantiles_from_MCMC()
