import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt

# Ensure the PK_PD folder exists inside SLE_results in Results
results_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Results', 'SLE_results', 'PK_PD'))
os.makedirs(results_folder, exist_ok=True)

# Map experiment names to doses
experiment_to_dose = {
    "IVdose_005_HV": "0.05 mg/kg",
    "IVdose_03_HV": "0.3 mg/kg",
    "IVdose_1_HV": "1 mg/kg",
    "IVdose_3_HV": "3 mg/kg",
    "IVdose_20_HV": "20 mg/kg",
    "SCdose_50_HV": "50 mg"
}

def plot_PK_PD_sim_with_data(params, sims, experiment, PK_data, PD_data, time_vectors, feature_to_plot_PK='PK_sim_skin', feature_to_plot_PD='PD_sim_skin'):
    if experiment not in PD_data:
        print(f"Skipping {experiment} as it is not present in both PK and PD data.")
        return None

    # Get the dose from the mapping
    dose = experiment_to_dose.get(experiment, "Unknown dose")
    figure_title = f"SLE, {dose} BIIB059"

    # Create a new figure
    fig, ax1 = plt.subplots()

    # Plot PK_sim on the left y-axis
    ax1.set_xlabel('Time [Hours]')
    ax1.set_ylabel('BIIB059 Plasma Concentration (µg/mL)', color='b')
    timepoints = time_vectors[experiment]

    # Simulate and plot PK data
    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
    feature_idx_PK = sims[experiment].feature_names.index(feature_to_plot_PK)
    ax1.plot(sims[experiment].time_vector, sims[experiment].feature_data[:, feature_idx_PK], 'b-')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis for PD_sim
    ax2 = ax1.twinx()
    ax2.set_ylabel('BDCA2 Expression % Change from Baseline', color='r')

    # Simulate and plot PD data
    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
    feature_idx_PD = sims[experiment].feature_names.index(feature_to_plot_PD)
    ax2.plot(sims[experiment].time_vector, sims[experiment].feature_data[:, feature_idx_PD], 'r-')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add a title with the updated experiment name and dose
    plt.title(figure_title)

    # Adjust layout
    plt.tight_layout()

    # Return the figure object for saving
    return fig

# Load the PK and PD data
with open("../../../Data/PK_data.json", "r") as pk_file:
    PK_data = json.load(pk_file)

with open("../../../Data/PD_data.json", "r") as pd_file:
    PD_data = json.load(pd_file)

# Load the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
first_model = sund.load_model("mPBPK_SLE_model")

# Create activities for the different doses
bodyweight = 70  # Bodyweight in kg
first_model_sims = {}

for experiment in PK_data:
    activity = sund.Activity(time_unit='h')
    input_data = PK_data[experiment]['input']['IV_in'] if 'IV_in' in PK_data[experiment]['input'] else PK_data[experiment]['input']['SC_in']
    activity.add_output(
        sund.PIECEWISE_CONSTANT,
        "IV_in" if 'IV_in' in PK_data[experiment]['input'] else "SC_in",
        t=input_data['t'],
        f=bodyweight * np.array(input_data['f']) if 'IV_in' in PK_data[experiment]['input'] else np.array(input_data['f'])
    )
    first_model_sims[experiment] = sund.Simulation(models=first_model, activities=activity, time_unit='h')

# Generate time vectors for each experiment
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 100, 1) for exp in PK_data}

# Define parameters (example parameters, replace with actual values)
params = [0.81995, 0.00867199496525978, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6927716105886019, 0.95, 0.7960584853135797, 0.2, 0.0096780180307827, 1.52, 1.82, 1.14185149185025, 14000.0]

# Plot and save each experiment
for experiment in PK_data:
    # Replace "_HV" with "_SLE" in the experiment name
    experiment_sle = experiment.replace("_HV", "_SLE")
    
    # Generate the plot
    fig = plot_PK_PD_sim_with_data(params, first_model_sims, experiment, PK_data, PD_data, time_vectors)
    if fig:  # Only save if a figure was returned
        # Save the figure with the updated name
        plot_path = os.path.join(results_folder, f"{experiment_sle}_PK_PD_simulation_low.png")
        fig.savefig(plot_path)
        print(f"Saved plot for {experiment_sle} to {plot_path}")