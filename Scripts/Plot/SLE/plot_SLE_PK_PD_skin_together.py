# Import necessary libraries
import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt

# Load SLE PK data
# with open("../../../Data/SLE_PK_data.json", "r") as f:
#     PK_data = json.load(f)

# Load SLE Validation PK data
with open("../../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load the best parameters for SLE
with open("../../../Results/Acceptable params/best_SLE_result_179_pdc_mm2.json", 'r') as f:
    params = np.array(json.load(f)['best_param'])

# Load acceptable parameters for SLE
with open("../../../Results/Acceptable params/acceptable_params_SLE_179_pdc_mm2.json", "r") as f:
    acceptable_params = json.load(f)

# Load the mPBPK_SLE_model
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
model = sund.load_model("mPBPK_SLE_model")

# Average bodyweight for SLE patients (cohort 8 in the phase 1 trial) 
bodyweight = 69

# Creating activity objects for each dose
# IV_20_SLE = sund.Activity(time_unit='h')
# IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=PK_data['IVdose_20_SLE']['input']['IV_in']['t'],  f=bodyweight * np.array(PK_data['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_SLE = sund.Activity(time_unit='h')
SC_50_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_50_SLE']['input']['SC_in']['f'])

SC_150_SLE = sund.Activity(time_unit='h')
SC_150_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_150_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_150_SLE']['input']['SC_in']['f'])

SC_450_SLE = sund.Activity(time_unit='h')
SC_450_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_450_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_450_SLE']['input']['SC_in']['f'])

# 'IVdose_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h')

# Creating simulation objects for each dose
model_sims = {
            'SCdose_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
            'SCdose_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
            'SCdose_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
            }

# Define the time vectors for each dose
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 2000, 1) for exp in PK_data}

# Define a function to plot PK and PD simulations in skin together
# Change BDCA2 baseline concentration in mPBPK_SLE_model.txt to create plots for different pDC densities in skin lesions
def plot_PK_PD_sim_with_data(params, acceptable_params, sims, PK_data, time_vectors, save_dir='../../../Results/SLE/Skin/PKPD'):
    os.makedirs(save_dir, exist_ok=True)

    labels = ['50 mg SC dose', '150 mg SC dose', '450 mg SC dose']

    # Loop through each experiment in PK_data
    for experiment, label in zip(PK_data, labels):
        fig, ax1 = plt.subplots(figsize=(8, 5))
        timepoints = time_vectors[experiment]

        # Create axes and labels for PK and PD
        ax1.set_xlabel('Time [Hours]')
        ax1.set_ylabel('Free Litifilimab Skin Concentration (µg/mL)', color='b')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Free BDCA2 Expression on pDCs (% Change from Baseline)', color='r')

        # Plot PK_sim_skin (left y-axis)
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for param in acceptable_params:
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=param, reset=True)
                y_sim = sims[experiment].feature_data[:, 2]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        ax1.fill_between(timepoints, y_min, y_max, color='b', alpha=0.3)


        sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
        y = sims[experiment].feature_data[:, 2]
        ax1.plot(timepoints, y, 'b-', label='PK simulation')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot PD_sim_skin (right y-axis) 
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for param in acceptable_params:
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=param, reset=True)
                y_sim = sims[experiment].feature_data[:, 3]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        ax2.fill_between(timepoints, y_min, y_max, color='r', alpha=0.3)


        y = sims[experiment].feature_data[:, 3]
        ax2.plot(timepoints, y, 'r-', label='PD simulation')
        ax2.tick_params(axis='y', labelcolor='r')

        # Legends and title
        ax1.legend(bbox_to_anchor=(0.95, 0.60))
        ax2.legend(bbox_to_anchor=(0.95, 0.50))
        plt.title(f"Simulation of a {label} for a patient with 179 pDCs/mm²")
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"{experiment}_PK_PD_sim_179_pDC_mm2.png")
        plt.savefig(save_path, dpi=600)
        plt.show()

# Plot PK and PD simulations in skin together
plot_PK_PD_sim_with_data(params, acceptable_params, model_sims, PK_data, time_vectors)
