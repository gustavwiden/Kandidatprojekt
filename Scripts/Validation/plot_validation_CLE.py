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

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data = json.load(f)

with open("../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params = json.load(f)

# Define a function to plot one PK_dataset
def plot_PK_dataset(PK_data, face_color='k'):
    plt.errorbar(PK_data['time'], PK_data['BIIB059_mean'], PK_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BIIB059 serum conc. (µg/ml)')

def plot_sim_PK(params, sim, timepoints, color='b', feature_to_plot='PK_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Install and load the model
sund.install_model('../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())
model = sund.load_model("mPBPK_SLE_model")

# Create activity objects for each dose
SC_50_CLE = sund.Activity(time_unit='h')
SC_50_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_CLE']['input']['SC_in']['t'],  f = PK_data['SCdose_50_CLE']['input']['SC_in']['f'])

SC_150_CLE = sund.Activity(time_unit='h')
SC_150_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_150_CLE']['input']['SC_in']['t'],  f = PK_data['SCdose_150_CLE']['input']['SC_in']['f'])

SC_450_CLE = sund.Activity(time_unit='h')
SC_450_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_450_CLE']['input']['SC_in']['t'],  f = PK_data['SCdose_450_CLE']['input']['SC_in']['f'])

# Create simulation objects for each dose
model_sims = {
    'SCdose_50_CLE': sund.Simulation(models=model, activities=SC_50_CLE, time_unit='h'),
    'SCdose_150_CLE': sund.Simulation(models=model, activities=SC_150_CLE, time_unit='h'),
    'SCdose_450_CLE': sund.Simulation(models=model, activities=SC_450_CLE, time_unit='h'),
}

# Create time vectors for each experiment
# The time vectors are created based on the maximum time in the PK_data for each experiment
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

# Optimal parameters for the mPBPK model when trained on HV data. Ksyn is adapted to fit the lower basline of BDCA2 in plasma in SLE patients.
SLE_params = [0.5982467918487137, 0.013501146489749132, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.75, 0.95, 0.7467544604963505, 0.2, 0.006287779429323163, 0.9621937056820449, 0.1, 5.539999999999999, 5.539999999999999, 2623.9999999999995]

def plot_model_uncertainty_with_validation_data(selected_params, acceptable_params, sims, PK_data, time_vectors, save_dir='../../Results/Validation', feature_to_plot='PK_sim'):
    os.makedirs(save_dir, exist_ok=True)

    colors = ['#6d65bf', '#6c5ce7', '#8c7ae6']

    markers = ['X', 'X', 'X']

    for i, (experiment, color) in enumerate(zip(PK_data.keys(), colors)):
        plt.figure()
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, np.inf)
        y_max = np.full_like(timepoints, -np.inf)

        # Calculate uncertainty range
        for params in acceptable_params:
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_sim = sims[experiment].feature_data[:, 0]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)

        # Plot selected parameter set
        sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        y_selected = sims[experiment].feature_data[:, 0]
        plt.plot(timepoints, y_selected, color=color, linewidth=2)

        # Plot experimental data
        marker = markers[i]
        plt.errorbar(
            PK_data[experiment]['time'],
            PK_data[experiment]['BIIB059_mean'],
            yerr=PK_data[experiment]['SEM'],
            fmt=marker,
            markersize=6,
            color=color,
            linestyle='None',
            capsize=3
        )

        plt.xlabel('Time [Hours]')
        plt.ylabel('BIIB059 Plasma Concentration (µg/ml)')
        plt.title(experiment)
        plt.tight_layout()

        # Save each figure with a unique name
        # save_path_svg = os.path.join(save_dir, f"PK_validation_with_{experiment}.svg")
        # plt.savefig(save_path_svg, format='svg')
        save_path_png = os.path.join(save_dir, f"PK_validation_with_{experiment}.png")
        plt.savefig(save_path_png, format='png', dpi=600)
        plt.close()

# Call the function as before
plot_model_uncertainty_with_validation_data(SLE_params, acceptable_params, model_sims, PK_data, time_vectors, save_dir='../../Results/Validation', feature_to_plot='PK_sim')

