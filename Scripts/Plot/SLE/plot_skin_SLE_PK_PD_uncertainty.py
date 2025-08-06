# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sund
import json
from scipy.stats import chi2
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
import sys
import csv
import random
import requests
import matplotlib.transforms as mtransforms

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# # Open the data file and read its contents
with open("../../../Data/SLE_PK_data_plotting.json", "r") as f:
    PK_data_phase_1 = json.load(f)

with open("../../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2A = json.load(f)

with open("../../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2B = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params = json.load(f)

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model")

# Creating activities for the different doses
bodyweight = 69 # Bodyweight for subject in kg

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


# Create simulation objects for each dose
model_sims_phase_1 = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h'),
}

model_sims_phase_2A = {
    'SCdose_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
    'SCdose_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
    'SCdose_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
}

model_sims_phase_2B = {
    'SCdose_50_CLE': sund.Simulation(models=model, activities=SC_50_CLE, time_unit='h'),
    'SCdose_150_CLE': sund.Simulation(models=model, activities=SC_150_CLE, time_unit='h'),
    'SCdose_450_CLE': sund.Simulation(models=model, activities=SC_450_CLE, time_unit='h'),
}

time_vectors_phase_1 = {exp: np.arange(-10, PK_data_phase_1[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_1}
time_vectors_phase_2A = {exp: np.arange(-10, PK_data_phase_2A[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2A}
time_vectors_phase_2B = {exp: np.arange(-10, PK_data_phase_2B[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2B}


def change_skin_specific_parameter(SLE_params, param_index, values):
    new_skin_params = []
    for v in values:
        original_params = SLE_params.copy()
        original_params[param_index] = v
        new_skin_params.append(original_params)
    return new_skin_params

SLE_params = [0.5982467918487137, 0.013501146489749132, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.75, 0.95, 0.7467544604963505, 0.2, 0.006287779429323163, 0.9621937056820449, 0.9621937056820449, 5.539999999999999, 5.539999999999999, 2623.9999999999995]

def plot_skin_PK_with_uncertainty(model, new_skin_params, SLE_params, acceptable_params, PK_data_dicts, time_vectors_dicts, sim_dicts, param_labels, save_dir = '../../../Results/SLE/Skin/PD/Parameter sensitivity/RCS'):

    linestyles = ['--', ':']
    colors = [ '#6d65bf', '#6c5ce7', '#8c7ae6']
    for idx, (dataset_name, PK_data) in enumerate(PK_data_dicts.items()):
        color = colors[idx % len(colors)]
        time_vectors = time_vectors_dicts[dataset_name]
        sims = sim_dicts[dataset_name]
        for experiment in PK_data:
            plt.figure(figsize=(8, 5))
            timepoints = time_vectors[experiment]

            for i, (params, label, ls) in enumerate(zip(new_skin_params, param_labels, linestyles)):
                try:
                    sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                    # y = sims[experiment].feature_data[:, 2] # PK plots
                    y = sims[experiment].feature_data[:, 3] # PD plots
                    plt.plot(timepoints, y, linestyle=ls, linewidth=2, color=color, label=label)
                except RuntimeError:
                    print(f"Simulation failed for {label} on {experiment}")
            
            y_min = np.full_like(timepoints, np.inf)
            y_max = np.full_like(timepoints, -np.inf)
            # Calculate uncertainty range
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

            # Plot uncertainty range
            plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)
            sims[experiment].simulate(time_vector=timepoints, parameter_values=SLE_params, reset=True)
            # y = sims[experiment].feature_data[:, 2] # PK plots
            y = sims[experiment].feature_data[:, 3] # PD plots
            plt.plot(timepoints, y, color=color, linewidth=2, label='RCS = 0.75')

            plt.xlabel('Time [Hours]')
            # plt.ylabel('BIIB059 Skin Concentration (µg/ml)')
            plt.ylabel('BDCA2 expression on pDCs (% change from baseline)')
            # plt.yscale('log')
            plt.title(f"{experiment} ({dataset_name})")
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{experiment}_skin_PD_RCS_uncertainty.png")
            plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)
            plt.close()

param_index = 10
parameter_values = [0.6, 0.9]
param_labels = ['RCS = 0.6', 'RCS = 0.9']

# param_index = 16
# parameter_values = [0.1, 10]
# param_labels = ['kdegs = 0.1 h-1', 'kdegs = 10 h-1']

# param_index = 18
# parameter_values = [0.5, 40]
# param_labels = ['kints = 0.5 h-1', 'kints = 40 h-1']


new_skin_params = change_skin_specific_parameter(SLE_params, param_index, parameter_values)

PK_data_dicts = {'phase_1': PK_data_phase_1, 'phase_2A': PK_data_phase_2A, 'phase_2B': PK_data_phase_2B}
time_vectors_dicts = { 'phase_1': time_vectors_phase_1, 'phase_2A': time_vectors_phase_2A, 'phase_2B': time_vectors_phase_2B}
sim_dicts = { 'phase_1': model_sims_phase_1, 'phase_2A': model_sims_phase_2A, 'phase_2B': model_sims_phase_2B }

plot_skin_PK_with_uncertainty(
    model=model,
    new_skin_params= new_skin_params,
    SLE_params=SLE_params,
    acceptable_params=acceptable_params,
    PK_data_dicts=PK_data_dicts,
    time_vectors_dicts=time_vectors_dicts,
    sim_dicts=sim_dicts,
    param_labels=param_labels,
)


