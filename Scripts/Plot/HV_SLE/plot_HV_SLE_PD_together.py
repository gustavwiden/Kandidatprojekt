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

with open("../../../Results/Acceptable params/acceptable_params.json", "r") as f:
    acceptable_params = json.load(f)

# Open the mPBPK_SLE_model.txt file and read its contents
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../../Data/SLE_PD_data_plotting.json", "r") as f:
    SLE_PD_data = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params_SLE = json.load(f)

def plot_all_doses_with_uncertainty(selected_params_HV, acceptable_params_HV, sims_HV,
                                    selected_params_SLE, acceptable_params_SLE, sims_SLE, SLE_PD_data, time_vectors,
                                    save_dir='../../../Results/HV_SLE/PD', feature_to_plot='PD_sim'):
    os.makedirs(save_dir, exist_ok=True)

    dose_colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'P', 'X']

    for i, experiment in enumerate(SLE_PD_data):
        plt.figure()
        timepoints = time_vectors[experiment]
        color = dose_colors[i % len(dose_colors)]
        marker = markers[i % len(markers)]

        # Simulate and collect successful results
        y_sims = []
        for params in acceptable_params_HV:
            try:
                sims_HV[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_sim = sims_HV[experiment].feature_data[:, 1]
                if np.any(np.isnan(y_sim)) or np.any(np.isinf(y_sim)):
                    continue
                y_sims.append(y_sim)
            except RuntimeError:
                continue

        # Stack successful sims into matrix
        if y_sims:
            y_matrix = np.stack(y_sims, axis=1)
            y_min = np.min(y_matrix, axis=1)
            y_max = np.max(y_matrix, axis=1)
            plt.fill_between(timepoints, y_min, y_max, color='lightgrey', alpha=0.3)


        # Plot optimal simulation
        sims_HV[experiment].simulate(time_vector=timepoints, parameter_values=selected_params_HV, reset=True)
        y_HV = sims_HV[experiment].feature_data[:, 1]
        plt.plot(timepoints, y_HV, color='black', linestyle='-', label='HV')

        
        y_sims = []
        for params in acceptable_params_SLE:
            try:
                sims_SLE[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_sim = sims_SLE[experiment].feature_data[:, 1] 
                if np.any(np.isnan(y_sim)) or np.any(np.isinf(y_sim)):
                    continue 
                y_sims.append(y_sim)
            except RuntimeError:
                continue

        # Stack successful sims into matrix
        if y_sims:
            y_matrix = np.stack(y_sims, axis=1)
            y_min = np.min(y_matrix, axis=1)
            y_max = np.max(y_matrix, axis=1)
            plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)


        # Plot optimal simulation
        sims_SLE[experiment].simulate(time_vector=timepoints, parameter_values=selected_params_SLE, reset=True)
        y_SLE = sims_SLE[experiment].feature_data[:, 1]
        plt.plot(timepoints, y_SLE, color=color, linestyle='-', label='SLE')

        ## ==== SLE DATA ONLY FOR 20mg/kg ====
        if experiment == 'IVdose_20_SLE':
            plt.errorbar(SLE_PD_data[experiment]['time'],
                         SLE_PD_data[experiment]['BDCA2_median'],
                         yerr=SLE_PD_data[experiment]['SEM'],
                         fmt=marker,
                         color=color,
                         linestyle='None',
                         label='SLE data')

        ## === Formatting ===
        plt.xlabel('Time [Hours]')
        plt.ylabel('BDCA2 expression on pDCs (% change from baseline)')
        plt.title(experiment)
        plt.legend()
        plt.tight_layout()

        save_path_svg = os.path.join(save_dir, f"{experiment}_vs_SLE_PD.svg")
        plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
        save_path_png = os.path.join(save_dir, f"{experiment}_vs_SLE_PD.png")
        plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=600)


## Setup of the model
# Install the model
sund.install_model('../../../Models/mPBPK_model.txt')
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_model")
SLE_model = sund.load_model("mPBPK_SLE_model")

# Creating activities for the different doses
bodyweight = 69 # Bodyweight for subject in kg

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



model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}
SLE_model_sims = {
    'IVdose_005_HV': sund.Simulation(models=SLE_model, activities=IV_005_HV, time_unit='h'),
    'IVdose_03_HV': sund.Simulation(models=SLE_model, activities=IV_03_HV, time_unit='h'),
    'IVdose_1_HV': sund.Simulation(models=SLE_model, activities=IV_1_HV, time_unit='h'),
    'IVdose_3_HV': sund.Simulation(models=SLE_model, activities=IV_3_HV, time_unit='h'),
    'IVdose_10_HV': sund.Simulation(models=SLE_model, activities=IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models=SLE_model, activities=IV_20_SLE, time_unit='h'),
    'SCdose_50_HV': sund.Simulation(models=SLE_model, activities=SC_50_HV, time_unit='h')
}


time_vectors = {exp: np.arange(-10, SLE_PD_data[exp]["time"][-1] + 2000, 1) for exp in SLE_PD_data}

params_HV = [0.6275806018256461, 0.012521665343092613, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.63, 0.95, 0.7965420036627042, 0.2, 0.005402247272171798, 67.02886707793152, 5.539999999999999, 2497.3959216508456]
params_SLE = [0.6275806018256461, 0.012521665343092613, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.63, 0.95, 0.7965420036627042, 0.2, 0.007233264815221344, 46.0, 831.4599999999999, 5.539999999999999, 231206.61954937642]

plot_all_doses_with_uncertainty(
    selected_params_HV=params_HV,
    acceptable_params_HV=acceptable_params,
    sims_HV=model_sims,
    selected_params_SLE=params_SLE,
    acceptable_params_SLE=acceptable_params_SLE,
    sims_SLE=SLE_model_sims,
    SLE_PD_data=SLE_PD_data,
    time_vectors=time_vectors
)

