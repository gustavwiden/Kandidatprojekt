# Import necessary libraries
import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt

# Load SLE PK data
with open("../../../Data/SLE_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load SLE PD data
with open("../../../Data/SLE_PD_data.json", "r") as f:
    PD_data = json.load(f)

# Load the final parameters for SLE
with open("../../../Models/final_parameters_SLE.json", "r") as f:   
    params = json.load(f)

# Load the mPBPK_SLE_model
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
model = sund.load_model("mPBPK_SLE_model")

# Average bodyweight for SLE patients (cohort 8 in the phase 1 trial) 
bodyweight = 69

# Creating activity objects for each dose
IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=PK_data['IVdose_20_SLE']['input']['IV_in']['t'],  f=bodyweight * np.array(PK_data['IVdose_20_SLE']['input']['IV_in']['f']))

# Creating simulation objects for each dose
model_sims = {'IVdose_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h')}

# Define the time vectors for each dose
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 2000, 1) for exp in PK_data}

# Define a function to plot PK and PD simulations in skin together
# Change BDCA2 baseline concentration in mPBPK_SLE_model.txt to create plots for different pDC densities in skin lesions
def plot_PK_PD_sim_with_data(params, sims, PK_data, PD_data, time_vectors, save_dir='../../../Results/SLE/Skin/PKPD'):
    os.makedirs(save_dir, exist_ok=True)

    # Loop through each experiment in PK_data
    for experiment in PK_data:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        timepoints = time_vectors[experiment]

        # Create axes and labels for PK and PD
        ax1.set_xlabel('Time [Hours]')
        ax1.set_ylabel('BIIB059 Skin Concentration (µg/mL)', color='b')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Free BDCA2 Expression % Change from Baseline', color='r')

        # Plot PK_sim_skin (left y-axis)
        sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
        y = sims[experiment].feature_data[:, 2]
        ax1.plot(timepoints, y, 'b-', label='PK simulation')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot PD_sim_skin (right y-axis) 
        y = sims[experiment].feature_data[:, 3]
        ax2.plot(timepoints, y, 'r-', label='PD simulation')
        ax2.tick_params(axis='y', labelcolor='r')

        # Legends and title
        ax1.legend(bbox_to_anchor=(0.95, 0.60))
        ax2.legend(bbox_to_anchor=(0.95, 0.50))
        plt.title(f"PK and PD Simulation for {experiment} with 179 pDCs/mm²")
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"{experiment}_PK_PD_sim_179_pDC_mm2.png")
        plt.savefig(save_path, dpi=600)
        plt.show()

# Plot PK and PD simulations in skin together
plot_PK_PD_sim_with_data(params, model_sims, PK_data, PD_data, time_vectors)
