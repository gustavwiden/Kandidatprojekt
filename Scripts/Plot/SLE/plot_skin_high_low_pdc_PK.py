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

# Open the mPBPK_model.txt file and read its contents
with open("../../../Models/high_pdc_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../../Data/SLE_PK_data_plotting.json", "r") as f:
    PK_data = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params = json.load(f)

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
sund.install_model('../../../Models/high_pdc_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model")
model_high_pdc = sund.load_model("high_pdc_model")

# Creating activities for the different doses
bodyweight = 69 # Bodyweight for subject in kg

IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data['SCdose_50_HV']['input']['SC_in']['f'])

model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

model_high_pdc_sims = {
    'IVdose_005_HV': sund.Simulation(models = model_high_pdc, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model_high_pdc, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model_high_pdc, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model_high_pdc, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model_high_pdc, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model_high_pdc, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model_high_pdc, activities = SC_50_HV, time_unit = 'h')
}


time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

original_params = [0.6275806018256461, 0.012521665343092613, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.63, 0.95, 0.7965420036627042, 0.2, 0.005585981224494457, 46.0, 3018.0000000000014, 5.539999999999999, 171190.7029110913]
high_params = [0.6275806018256461, 0.012521665343092613, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.63, 0.95, 0.7965420036627042, 0.2, 0.005585981224494457, 46.0, 16882, 5.539999999999999, 171190.7029110913]

all_params = [original_params, high_params]
all_sims = [model_sims, model_high_pdc_sims]

def plot_all_doses_with_high_low_pdc_uncertainty_gradient(selected_param_sets, acceptable_params, sim_sets, PK_data, time_vectors, save_dir='../../../Results/SLE/Skin/PK', feature_to_plot='PK_sim_skin'):
    os.makedirs(save_dir, exist_ok=True)

    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']

    for i, experiment in enumerate(PK_data):
        plt.figure(figsize=(8, 5))
        timepoints = time_vectors[experiment]
        color = colors[i % len(colors)]
        y_min = np.full_like(timepoints, np.inf)
        y_max = np.full_like(timepoints, -np.inf)

        for params in acceptable_params:
            try:
                model_sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_sim = model_sims[experiment].feature_data[:, 2]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)


        # Plot each model simulation with selected param set (same color)
        for i, label in enumerate(['32 pDCs/mm2', '179 pDCs/mm2']):
            sim = sim_sets[i][experiment]
            params = selected_param_sets[i]
            try:
                sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_main = sim.feature_data[:, 2]
                plt.plot(timepoints, y_main, label=label, color=color, linestyle=['-', '--'][i])
            except RuntimeError:
                print(f"Simulation failed for {label} on {experiment}")

        plt.xlabel('Time [Hours]')
        plt.ylabel('BIIB059 Skin Concentration (µg/ml)')
        plt.title(experiment)
        plt.legend()
        plt.tight_layout()

        save_path_svg = os.path.join(save_dir, f"{experiment}_PK_skin_high_low_pdc.svg")
        plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
        save_path_png = os.path.join(save_dir, f"{experiment}_PK_skin_high_low_pdc.png")
        plt.savefig(save_path_png, format='png', bbox_inches='tight', dpi=600)



plot_all_doses_with_high_low_pdc_uncertainty_gradient(
    selected_param_sets=[original_params, high_params],
    acceptable_params=acceptable_params,
    sim_sets=[model_sims, model_high_pdc_sims],
    PK_data=PK_data,
    time_vectors=time_vectors
)


