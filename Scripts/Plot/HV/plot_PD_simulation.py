# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sund
import json

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the PD data file and read its contents
with open("../../../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

# Load acceptable parameters for mPBPK-model from PL
acceptable_params_PL = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()

# Load final parameters for mPBPK-model form optimization
with open("../../../Models/final_parameters.json", "r") as f:
    params = json.load(f)

# Define a function to plot all doses with uncertainty in different figures
def plot_all_doses_with_uncertainty(selected_params, acceptable_params, sims, PD_data, time_vectors, save_dir='../../../Results/HV/PD'):
    os.makedirs(save_dir, exist_ok=True)

    # Define colors and markers for each dose
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'P', 'X']
    doses = ['0.05 mg/kg IV Dose', '0.3 mg/kg IV Dose', '1 mg/kg IV Dose', '3 mg/kg IV Dose', '20 mg/kg IV Dose', '50 mg SC Dose']

    #Loop through each experiment
    for i, (experiment, color, dose) in enumerate(zip(PD_data.keys(), colors, doses)):
        plt.figure(figsize=(10, 8))
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params:
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_sim = sims[experiment].feature_data[:, 1]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3, label='Uncertainty')

        # Plot selected parameter set
        sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        y_selected = sims[experiment].feature_data[:, 1]
        plt.plot(timepoints, y_selected, color=color, label='Simulation', linewidth=3)

        # Plot experimental data
        marker = markers[i]
        plt.errorbar(
            PD_data[experiment]['time'],
            PD_data[experiment]['BDCA2_median'],
            yerr=PD_data[experiment]['SEM'],
            marker=marker,
            markersize=8,
            color=color,
            linestyle='None',
            capsize=4,
            elinewidth=2,
            label='Data'
        )

        # Set labels
        plt.xlabel('Time [Hours]', fontsize=18)
        plt.ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=18)
        plt.title('PD Simulation in Plasma of Healthy Volunteer', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(title=f'{dose}', title_fontsize=18, fontsize=16, loc='lower right')

        if experiment == 'IVdose_20_HV':
            plt.xlim(-200, 7000)
        elif experiment == 'IVdose_3_HV':
            plt.xlim(-140, 4800)
        else:
            plt.xlim(-90, 3000)

        if experiment == 'IVdose_03_HV':
            plt.ylim(-118, 39)
        else:
            plt.ylim(-118, 19)

        plt.tight_layout()

        # Save the figure
        save_path_png = os.path.join(save_dir, f"{experiment}_PD_plot.png")
        plt.savefig(save_path_png, format='png', dpi=600)
        plt.close()

## Setup of the model
# Install the model
sund.install_model('../../../Models/mPBPK_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_model")

# Average bodyweight for healthy volunteers (HV) (cohort 1-7 in the phase 1 trial)
bodyweight = 73

# Creating activity objects for each dose
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output('piecewise_constant', "SC_in",  t = PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PD_data['SCdose_50_HV']['input']['SC_in']['f'])

# Creating simulation objects for each dose
model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

# Define time vectors for each dose
time_vectors = {exp: np.arange(-400, PD_data[exp]["time"][-1] + 5000, 1) for exp in PD_data}

# Plot all doses with uncertainty
plot_all_doses_with_uncertainty(params, acceptable_params_PL, model_sims, PD_data, time_vectors)
