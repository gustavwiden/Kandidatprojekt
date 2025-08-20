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

# Load the acceptable parameters for HV
with open("../../../Results/Acceptable params/acceptable_params.json", "r") as f:
    acceptable_params = json.load(f)

with open("../../../Models/final_parameters.json", "r") as f:
    params_HV = json.load(f)

# Open the mPBPK_SLE_model.txt file and read its contents
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Load the SLE PD data for plotting
with open("../../../Data/SLE_PD_data_plotting.json", "r") as f:
    SLE_PD_data = json.load(f)

# Load the acceptable parameters for SLE
with open("../../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params_SLE = json.load(f)

with open("../../../Models/final_parameters_SLE.json", "r") as f:
    params_SLE = json.load(f)

# Define a function to plot all doses with uncertainty
def plot_all_doses_with_uncertainty(params_HV, acceptable_params_HV, sims_HV, params_SLE, acceptable_params_SLE, sims_SLE, SLE_PD_data, time_vectors, save_dir='../../../Results/SLE/Plasma/PD'):
    os.makedirs(save_dir, exist_ok=True)

    # Colors and markers for different doses
    dose_colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'P', 'X']

    # Loop through each experiment in SLE_PD_data
    for i, experiment in enumerate(SLE_PD_data):
        plt.figure()
        timepoints = time_vectors[experiment]
        color = dose_colors[i % len(dose_colors)]
        marker = markers[i % len(markers)]

        # HV
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params_HV:
            try:
                sims_HV[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_HV = sims_HV[experiment].feature_data[:, 1]
                y_min = np.minimum(y_min, y_HV)
                y_max = np.maximum(y_max, y_HV)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color='lightgrey', alpha=0.3)

        # Plot optimal parameters
        sims_HV[experiment].simulate(time_vector=timepoints, parameter_values=params_HV, reset=True)
        y_HV = sims_HV[experiment].feature_data[:, 1]
        plt.plot(timepoints, y_HV, color='black', linestyle='-', label='HV')

        
       # SLE
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params_SLE:
            try:
                sims_SLE[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_SLE = sims_SLE[experiment].feature_data[:, 1]
                y_min = np.minimum(y_min, y_SLE)
                y_max = np.maximum(y_max, y_SLE)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)

        # Plot optimal simulation
        sims_SLE[experiment].simulate(time_vector=timepoints, parameter_values=params_SLE, reset=True)
        y_SLE = sims_SLE[experiment].feature_data[:, 1]
        plt.plot(timepoints, y_SLE, color=color, linestyle='-', label='SLE')

        # Plot the only available SLE data set
        if experiment == 'IVdose_20_SLE':
            plt.errorbar(SLE_PD_data[experiment]['time'],
                         SLE_PD_data[experiment]['BDCA2_median'],
                         yerr=SLE_PD_data[experiment]['SEM'],
                         fmt=marker,
                         color=color,
                         linestyle='None',
                         label='SLE data')

        # Set plot labels and legend
        plt.xlabel('Time [Hours]')
        plt.ylabel('Total BDCA2 Expression on pDCs (% Change from Baseline)')
        plt.title(experiment)
        plt.legend()
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"{experiment}_vs_SLE_PD.png")
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)


# Install the models
sund.install_model('../../../Models/mPBPK_model.txt')
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model objects
model = sund.load_model("mPBPK_model")
SLE_model = sund.load_model("mPBPK_SLE_model")

# Average bodyweight for SLE patients (cohort 8 in the phase 1 trial)
# Since HV plots are only for comparison, not to fit PD data, the same bodyweight is used
bodyweight = 69

# Creating activity objects for each dose
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = SLE_PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = SLE_PD_data['SCdose_50_HV']['input']['SC_in']['f'])

# Creating simulation objects for each dose in HV
HV_model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

# Creating simulation objects for each dose in SLE
SLE_model_sims = {
    'IVdose_005_HV': sund.Simulation(models=SLE_model, activities=IV_005_HV, time_unit='h'),
    'IVdose_03_HV': sund.Simulation(models=SLE_model, activities=IV_03_HV, time_unit='h'),
    'IVdose_1_HV': sund.Simulation(models=SLE_model, activities=IV_1_HV, time_unit='h'),
    'IVdose_3_HV': sund.Simulation(models=SLE_model, activities=IV_3_HV, time_unit='h'),
    'IVdose_10_HV': sund.Simulation(models=SLE_model, activities=IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models=SLE_model, activities=IV_20_SLE, time_unit='h'),
    'SCdose_50_HV': sund.Simulation(models=SLE_model, activities=SC_50_HV, time_unit='h')
}

# Define the time vectors for each experiment
time_vectors = {exp: np.arange(-10, SLE_PD_data[exp]["time"][-1] + 2000, 1) for exp in SLE_PD_data}

# Call the function to plot all doses with uncertainty
plot_all_doses_with_uncertainty(
    params_HV=params_HV,
    acceptable_params_HV=acceptable_params,
    sims_HV=HV_model_sims,
    params_SLE=params_SLE,
    acceptable_params_SLE=acceptable_params_SLE,
    sims_SLE=SLE_model_sims,
    SLE_PD_data=SLE_PD_data,
    time_vectors=time_vectors
)

