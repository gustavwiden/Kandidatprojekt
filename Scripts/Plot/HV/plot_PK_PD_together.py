import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt

# Ensure the PK_PD folder exists inside HV_results in Results
results_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Results', 'HV_results', 'PK_PD'))
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

def plot_PK_PD_sim_with_data(params, sims, experiment, PK_data, PD_data, time_vectors, feature_to_plot_PK='PK_sim', feature_to_plot_PD='PD_sim'):
    if experiment not in PD_data:
        print(f"Skipping {experiment} as it is not present in both PK and PD data.")
        return None

    # Get the dose from the mapping
    dose = experiment_to_dose.get(experiment, "Unknown dose")
    figure_title = f"HV, {dose} BIIB059"

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
with open("../Data/PK_data.json", "r") as pk_file:
    PK_data = json.load(pk_file)

with open("../Data/PD_data.json", "r") as pd_file:
    PD_data = json.load(pd_file)

# Load the model
sund.install_model('../Models/mPBPK_model.txt')
first_model = sund.load_model("mPBPK_model")

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
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 2000, 1) for exp in PK_data}

# Define parameters (example parameters, replace with actual values)
params = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.77, 0.95, 0.605, 0.2, 5.51, 14.15, 0.28, 2.12e-05, 2.5, 0.525, 4.08e-05]

# Plot and save each experiment
for experiment in PK_data:
    fig = plot_PK_PD_sim_with_data(params, first_model_sims, experiment, PK_data, PD_data, time_vectors)
    if fig:  # Only save if a figure was returned
        plot_path = os.path.join(results_folder, f"{experiment}_PK_PD_simulation.png")
        fig.savefig(plot_path)
        print(f"Saved plot for {experiment} to {plot_path}")