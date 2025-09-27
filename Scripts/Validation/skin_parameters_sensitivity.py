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
with open("../../Models/mPBPK_SLE_model_32_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

# Load SLE PK data from phase 1
with open("../../Data/SLE_PK_data_plotting.json", "r") as f:
    PK_data_phase_1 = json.load(f)

# Load SLE PK data from phase 2A
with open("../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2A = json.load(f)

# Load CLE PK data from phase 2B
with open("../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2B = json.load(f)

# Load acceptable parameters for mPBPK_SLE_model
with open("../../Results/Acceptable params/acceptable_params_SLE_pre_skin_investigation.json", "r") as f:
    acceptable_params = json.load(f)

# Load best parameters found for the mPBPK_SLE_model yet
with open("../../Results/Acceptable params/best_SLE_result_pre_skin_investigation.json", "r") as f:
    best_data = json.load(f)
    best_params = np.array(best_data['best_param'])

# Install the model
sund.install_model('../../Models/mPBPK_SLE_model_32_pdc_mm2.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model_32_pdc_mm2")

# Creating activity objects for each dose

# Average bodyweight for SLE patients (cohort 8 in the phase 1 trial)
# Phase 2 only included SC doses which size are independent of bodyweight
bodyweight = 69

# Creating activities for the different doses
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output('piecewise_constant', "IV_in",  t = PK_data_phase_1['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output('piecewise_constant', "IV_in",  t = PK_data_phase_1['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output('piecewise_constant', "IV_in",  t = PK_data_phase_1['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output('piecewise_constant', "IV_in",  t = PK_data_phase_1['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output('piecewise_constant', "IV_in",  t = PK_data_phase_1['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output('piecewise_constant', "IV_in",  t = PK_data_phase_1['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output('piecewise_constant', "SC_in",  t = PK_data_phase_1['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data_phase_1['SCdose_50_HV']['input']['SC_in']['f'])

SC_50_SLE = sund.Activity(time_unit='h')
SC_50_SLE.add_output('piecewise_constant', "SC_in",  t = PK_data_phase_2A['SCdose_50_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_50_SLE']['input']['SC_in']['f'])

SC_150_SLE = sund.Activity(time_unit='h')
SC_150_SLE.add_output('piecewise_constant', "SC_in",  t = PK_data_phase_2A['SCdose_150_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_150_SLE']['input']['SC_in']['f'])

SC_450_SLE = sund.Activity(time_unit='h')
SC_450_SLE.add_output('piecewise_constant', "SC_in",  t = PK_data_phase_2A['SCdose_450_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_450_SLE']['input']['SC_in']['f'])

SC_50_CLE = sund.Activity(time_unit='h')
SC_50_CLE.add_output('piecewise_constant', "SC_in",  t = PK_data_phase_2B['SCdose_50_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_50_CLE']['input']['SC_in']['f'])

SC_150_CLE = sund.Activity(time_unit='h')
SC_150_CLE.add_output('piecewise_constant', "SC_in",  t = PK_data_phase_2B['SCdose_150_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_150_CLE']['input']['SC_in']['f'])

SC_450_CLE = sund.Activity(time_unit='h')
SC_450_CLE.add_output('piecewise_constant', "SC_in",  t = PK_data_phase_2B['SCdose_450_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_450_CLE']['input']['SC_in']['f'])

# Create simulation objects for each dose in phase 1
model_sims_phase_1 = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h'),
}

# Create simulation objects for each dose in phase 2A
model_sims_phase_2A = {
    'SCdose_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
    'SCdose_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
    'SCdose_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
}

# Create simulation objects for each dose in phase 2B
model_sims_phase_2B = {
    'SCdose_50_CLE': sund.Simulation(models=model, activities=SC_50_CLE, time_unit='h'),
    'SCdose_150_CLE': sund.Simulation(models=model, activities=SC_150_CLE, time_unit='h'),
    'SCdose_450_CLE': sund.Simulation(models=model, activities=SC_450_CLE, time_unit='h'),
}

# Define time vectors for doses in each trial
time_vectors_phase_1 = {exp: np.arange(-10, PK_data_phase_1[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_1}
time_vectors_phase_2A = {exp: np.arange(-10, PK_data_phase_2A[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2A}
time_vectors_phase_2B = {exp: np.arange(-10, PK_data_phase_2B[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2B}

# Define a function to change the value of skin-specific parameters
def change_skin_specific_parameters(best_params, param_indices, param_values, param_labels):
    new_skin_params = []
    for l in range(len(param_labels)):
        best_params_copy = best_params.copy()
        for i, index in enumerate(param_indices):
            best_params_copy[index] = param_values[i][l]
        new_skin_params.append(best_params_copy)
    return new_skin_params

# Define a function to plot the influence of skin-specific parameters on PK or PD simulations
def plot_skin_PK_PD_with_uncertainty(new_skin_params, best_params, acceptable_params, PK_data_dicts, time_vectors_dicts, sim_dicts, param_labels, param_name, save_dir = '../../Results/SLE/Skin/PK/Parameter sensitivity/kints'):
    os.makedirs(save_dir, exist_ok=True)

    # Linestyles for different parameter sets
    linestyles = ['-','--',':','-.']

    # Colors for different doses
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']

    # Dose labels
    dose_labels = ['0.05 mg/kg IV', '0.3 mg/kg IV', '1 mg/kg IV', '3 mg/kg IV', '10 mg/kg IV', '20 mg/kg IV', '50 mg SC']

    # Loop through each dataset
    for dataset_name, PK_data in PK_data_dicts.items():
        time_vectors = time_vectors_dicts[dataset_name]
        sims = sim_dicts[dataset_name]

        # Create a figure for each dose
        for experiment, color, dose_label in zip(PK_data, colors, dose_labels):
            plt.figure(figsize=(12, 8))
            timepoints = time_vectors[experiment]
            
            # Calculate uncertainty range for the original parameters
            y_min = np.full_like(timepoints, 10000)
            y_max = np.full_like(timepoints, -10000)
            for params in acceptable_params:
                try:
                    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                    y = sims[experiment].feature_data[:, 2] # PK plots
                    # y = sims[experiment].feature_data[:, 3] # PD plots
                    y_min = np.minimum(y_min, y)
                    y_max = np.maximum(y_max, y)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {experiment}")
                    else:
                        raise e

            # Plot uncertainty range
            plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)
            sims[experiment].simulate(time_vector=timepoints, parameter_values=best_params, reset=True) 

            # Plot simulations for each new skin parameter
            for params, param_label, ls in zip(new_skin_params, param_labels, linestyles):
                try:
                    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                    y = sims[experiment].feature_data[:, 2] # PK plots
                    # y = sims[experiment].feature_data[:, 3] # PD plots
                    plt.plot(timepoints, y, linestyle=ls, linewidth=3, color=color, label=param_label)
                except RuntimeError:
                    print(f"Simulation failed for {param_label} on {experiment}")         

            # Set labels, title and legends
            plt.xlabel('Time [Hours]', fontsize=18)
            plt.ylabel('BIIB059 Skin Concentration [µg/ml]', fontsize=18)
            # plt.ylabel('Free BDCA2 expression on pDCs [% Change]', fontsize=18)
            plt.yscale('log')
            plt.title(f"Sensitivity Analysis of the {param_name} Parameter for a {dose_label} Dose", fontsize=22)
            # if experiment == 'IVdose_20_SLE':
            #     plt.legend(fontsize='14', loc='lower left', bbox_to_anchor=(0.07, 0.07))
            # else:
            #     plt.legend(fontsize='14', loc='lower right')
            plt.legend(fontsize='14', loc='upper right')
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(save_dir, f"{experiment}_skin_PK_kints_sensitivity.png")
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)
            plt.close()

# Simulate the models sensitivity to changes in RCS
# param_name = r'$\sigma_s$'
# param_indices = [10]
# param_values = [[0.95, 0.5, 0.75]]
# param_labels = [r'$\boldsymbol{\sigma}_{\mathbf{s}} = \mathbf{0.95}$', r'$\sigma_s = 0.5$', r'$\sigma_s = 0.75$']

# Simulate the models sensitivity to changes in kdegs
# param_name = r'$k_{\mathrm{deg}_s}$'
# param_indices = [16]
# param_values = [[0.96, 0.02, 0.5]]
# param_labels = [r'$\mathbf{k}_{\mathbf{deg}_\mathbf{s}} = \mathbf{0.96}\ \mathbf{h}^{-1}$', r'$k_{\mathrm{deg}_s} = 0.02\ \mathrm{h}^{-1}$', r'$k_{\mathrm{deg}_s} = 0.5\ \mathrm{h}^{-1}$']

# Simulate the models sensitivity to changes in kints
param_name = r'$k_{\mathrm{int}_s}$'
param_indices = [18]
param_values = [[5.54, 0.5, 40]]
param_labels = [r'$\mathbf{k}_{\mathbf{int}_\mathbf{s}} = \mathbf{5.54}\ \mathbf{h}^{-1}$', r'$k_{\mathrm{int}_s} = 0.5\ \mathrm{h}^{-1}$', r'$k_{\mathrm{int}_s} = 40\ \mathrm{h}^{-1}$']

# Simulate the models sensitivity to changes in kdegs with the new RCS value of 0.8
# param_name = r'$k_{\mathrm{deg}_s}$'
# param_indices = [10, 16]
# param_values = [[0.95, 0.8, 0.8, 0.8], [0.96, 0.2, 0.1, 0.05]]
# param_labels = [r'$\mathbf{k}_{\mathbf{deg}_\mathbf{s}} = \mathbf{0.96}\ \mathbf{h}^{-1},\ \boldsymbol{\sigma}_{\mathbf{s}} = \mathbf{0.95}$', r'$k_{\mathrm{deg}_s} = 0.2\ \mathrm{h}^{-1}, \sigma_s = 0.8$', r'$k_{\mathrm{deg}_s} = 0.1\ \mathrm{h}^{-1}, \sigma_s = 0.8$', r'$k_{\mathrm{deg}_s} = 0.05\ \mathrm{h}^{-1}, \sigma_s = 0.8$']


# Create new parameter sets with the changed skin-specific parameter
new_skin_params = change_skin_specific_parameters(best_params, param_indices, param_values, param_labels)

# Create dictionaries for PK data, time vectors and simulation objects with only phase 1 (used for finding optimal skin-parameter values)
PK_data_dicts = {'phase_1': PK_data_phase_1}
time_vectors_dicts = { 'phase_1': time_vectors_phase_1}
sim_dicts = { 'phase_1': model_sims_phase_1}

# Create dictionaries for PK data, time vectors and simulation objects with phase 1 and 2AB (used to plot validations once optimal values for skin-parameters have been found)
# PK_data_dicts = {'phase_1': PK_data_phase_1, 'phase_2A': PK_data_phase_2A, 'phase_2B': PK_data_phase_2B}
# time_vectors_dicts = { 'phase_1': time_vectors_phase_1, 'phase_2A': time_vectors_phase_2A, 'phase_2B': time_vectors_phase_2B}
# sim_dicts = { 'phase_1': model_sims_phase_1, 'phase_2A': model_sims_phase_2A, 'phase_2B': model_sims_phase_2B }

# Plot the influence of skin-specific parameters on PK or PD simulations
plot_skin_PK_PD_with_uncertainty(
    new_skin_params=new_skin_params,
    best_params=best_params,
    acceptable_params=acceptable_params,
    PK_data_dicts=PK_data_dicts,
    time_vectors_dicts=time_vectors_dicts,
    sim_dicts=sim_dicts,
    param_labels=param_labels,
    param_name=param_name
)


