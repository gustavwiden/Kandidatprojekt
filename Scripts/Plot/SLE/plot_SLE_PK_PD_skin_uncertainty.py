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
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Load SLE PK data from phase 1
with open("../../../Data/SLE_PK_data_plotting.json", "r") as f:
    PK_data_phase_1 = json.load(f)

# Load SLE PK data from phase 2A
with open("../../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2A = json.load(f)

# Load CLE PK data from phase 2B
with open("../../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2B = json.load(f)

# Load acceptable parameters for mPBPK_SLE_model
with open("../../../Results/Acceptable params/acceptable_params_SLE_pre_skin_investigation.json", "r") as f:
    acceptable_params = json.load(f)

# Load best parameters found for the mPBPK_SLE_model yet
with open("../../../Results/Acceptable params/best_SLE_result_pre_skin_investigation.json", "r") as f:
    best_data = json.load(f)
    best_params = np.array(best_data['best_param'])

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model")

# Creating activity objects for each dose

# Average bodyweight for SLE patients (cohort 8 in the phase 1 trial)
# Phase 2 only included SC doses which size are independent of bodyweight
bodyweight = 69

# Creating activities for the different doses
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_1['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data_phase_1['SCdose_50_HV']['input']['SC_in']['f'])

SC_50_SLE = sund.Activity(time_unit='h')
SC_50_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2A['SCdose_50_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_50_SLE']['input']['SC_in']['f'])

SC_150_SLE = sund.Activity(time_unit='h')
SC_150_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2A['SCdose_150_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_150_SLE']['input']['SC_in']['f'])

SC_450_SLE = sund.Activity(time_unit='h')
SC_450_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2A['SCdose_450_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_450_SLE']['input']['SC_in']['f'])

SC_50_CLE = sund.Activity(time_unit='h')
SC_50_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2B['SCdose_50_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_50_CLE']['input']['SC_in']['f'])

SC_150_CLE = sund.Activity(time_unit='h')
SC_150_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2B['SCdose_150_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_150_CLE']['input']['SC_in']['f'])

SC_450_CLE = sund.Activity(time_unit='h')
SC_450_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2B['SCdose_450_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_450_CLE']['input']['SC_in']['f'])

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
def plot_skin_PK_PD_with_uncertainty(model, new_skin_params, best_params, acceptable_params, PK_data_dicts, time_vectors_dicts, sim_dicts, param_labels, save_dir = '../../../Results/SLE/Skin/PD/Parameter sensitivity/Combination'):
    os.makedirs(save_dir, exist_ok=True)

    # Linestyles for different parameter sets
    linestyles = ['--', ':', '-.']

    # Colors for different doses
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf', '#6d65bf', '#6c5ce7', '#8c7ae6', '#6d65bf', '#6c5ce7', '#8c7ae6']

    # Loop through each dataset
    for idx, (dataset_name, PK_data) in enumerate(PK_data_dicts.items()):
        color = colors[idx % len(colors)]
        time_vectors = time_vectors_dicts[dataset_name]
        sims = sim_dicts[dataset_name]

        # Create a figure for each dose
        for experiment in PK_data:
            plt.figure(figsize=(8, 5))
            timepoints = time_vectors[experiment]

            # Plot simulations for each new skin parameter
            for i, (params, label, ls) in enumerate(zip(new_skin_params, param_labels, linestyles)):
                try:
                    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                    # y = sims[experiment].feature_data[:, 2] # PK plots
                    y = sims[experiment].feature_data[:, 3] # PD plots
                    plt.plot(timepoints, y, linestyle=ls, linewidth=2, color=color, label=label)
                except RuntimeError:
                    print(f"Simulation failed for {label} on {experiment}")
            
            # Calculate uncertainty range for the original parameters
            y_min = np.full_like(timepoints, np.inf)
            y_max = np.full_like(timepoints, -np.inf)
            for params in acceptable_params:
                try:
                    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                    # y = sims[experiment].feature_data[:, 2] # PK plots
                    y = sims[experiment].feature_data[:, 3] # PD plots
                    y_min = np.minimum(y_min, y)
                    y_max = np.maximum(y_max, y)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {experiment}")
                    else:
                        raise e

            # Plot uncertainty range and original parameter set
            plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)
            sims[experiment].simulate(time_vector=timepoints, parameter_values=best_params, reset=True)          
            # y = sims[experiment].feature_data[:, 2] # PK plots
            y = sims[experiment].feature_data[:, 3] # PD plots
            plt.plot(timepoints, y, color=color, linewidth=2, label='kdegs = 0.96')

            # Set labels, title and legends
            plt.xlabel('Time [Hours]')
            # plt.ylabel('BIIB059 Skin Concentration (µg/ml)')
            plt.ylabel('BDCA2 expression on pDCs (% change from baseline)')
            # plt.yscale('log')
            plt.title(f"{experiment} ({dataset_name})")
            plt.legend()
            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(save_dir, f"{experiment}_skin_PD_kdegs_RCS_combination.png")
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)
            plt.close()

# Simulate the models sensitivity to changes in RCS
# param_indices = [10]
# param_values = [[0.5, 0.75]]
# param_labels = ['RCS = 0.5', 'RCS = 0.75']

# Simulate the models sensitivity to changes in kdegs
# param_indices = [16]
# param_values = [[0.02, 0.5]]
# param_labels = ['kdegs = 0.02 h-1', 'kdegs = 0.5 h-1']

# Simulate the models sensitivity to changes in kints
# param_indices = [18]
# param_values = [[0.5, 40]]
# param_labels = ['kints = 0.5 h-1', 'kints = 40 h-1']

# Simulate the models sensitivity to changes in kints and RCS simultaneously
param_indices = [10, 16]
param_values = [[0.8, 0.8, 0.8], [0.2, 0.1, 0.05]]
param_labels = ['kdegs = 0.2', 'kdegs = 0.1', 'kdegs = 0.05']


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
    model=model,
    new_skin_params= new_skin_params,
    best_params=best_params,
    acceptable_params=acceptable_params,
    PK_data_dicts=PK_data_dicts,
    time_vectors_dicts=time_vectors_dicts,
    sim_dicts=sim_dicts,
    param_labels=param_labels,
)


