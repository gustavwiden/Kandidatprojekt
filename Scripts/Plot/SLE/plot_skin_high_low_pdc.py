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

# Open the mPBPK_model.txt file and read its contents
with open("../../../Models/low_pdc_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../../Data/SLE_PK_data_plotting.json", "r") as f:
    PK_data = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE_high.json", "r") as f:
    acceptable_params_high = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE_low.json", "r") as f:
    acceptable_params_low = json.load(f)

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
sund.install_model('../../../Models/high_pdc_model.txt')
sund.install_model('../../../Models/low_pdc_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model")
model_high_pdc = sund.load_model("high_pdc_model")
model_low_pdc = sund.load_model("low_pdc_model")

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

model_low_pdc_sims = {
    'IVdose_005_HV': sund.Simulation(models = model_low_pdc, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model_low_pdc, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model_low_pdc, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model_low_pdc, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model_low_pdc, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model_low_pdc, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model_low_pdc, activities = SC_50_HV, time_unit = 'h')
}

time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

original_params = [0.70167507023512, 0.010970491553609206, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.5908548614616957, 0.95, 0.7272247648651022, 0.2, 0.008418983535737234, 7.23, 66.97, 0.08300000082999998, 14123.510378662331, 80063718.67276345]
high_params = [0.70167507023512, 0.010970491553609206, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.5908548614616957, 0.95, 0.7272247648651022, 0.2, 0.007453322981954208, 7.23, 827.6300000000002, 0.08300154360127418, 14123.510378662331, 80063718.67276345]
low_params = [0.70167507023512, 0.010970491553609206, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.5908548614616957, 0.95, 0.7272247648651022, 0.2, 0.008579120035963624, 7.23, 20.930000000000003, 0.08299999999999998, 14123.510378662331, 80063718.67276345]

all_params = [original_params, high_params, low_params]
all_sims = [model_sims, model_high_pdc_sims, model_low_pdc_sims]

import matplotlib.colors as mcolors

def adjust_color_brightness(color, factor):
    """Brighten (>1) or darken (<1) a hex color."""
    rgb = mcolors.to_rgb(color)
    adjusted = [min(1, max(0, c * factor)) for c in rgb]
    return adjusted

def plot_gradient_fill_between(ax, x, y1, y2, base_color, n_shades=100):
    """
    Plot a vertical gradient fill between two curves using many thin slices.
    """
    from matplotlib.colors import to_rgb
    c_base = np.array(to_rgb(base_color))
    c_dark = c_base * 0.6
    c_light = np.clip(c_base * 1.4, 0, 1)

    for i in range(n_shades):
        alpha = (i + 0.5) / n_shades
        blend = alpha
        if blend < 0.5:
            factor = blend * 2
            color = c_dark + (c_base - c_dark) * factor
        else:
            factor = (blend - 0.5) * 2
            color = c_base + (c_light - c_base) * factor

        y_lower = y1 + (y2 - y1) * (i / n_shades)
        y_upper = y1 + (y2 - y1) * ((i + 1) / n_shades)

        ax.fill_between(x, y_lower, y_upper, color=color, linewidth=0)


def plot_all_doses_with_high_low_pdc_uncertainty_gradient(selected_param_sets, acceptable_param_sets, sim_sets, PK_data, time_vectors, save_dir='../../../Results/SLE/Skin/PK', feature_to_plot='PK_sim_skin'):
    os.makedirs(save_dir, exist_ok=True)

    # Your original dose-specific colors
    dose_colors = {
        'IVdose_005_HV': '#1b7837',
        'IVdose_03_HV': '#01947b',
        'IVdose_1_HV': '#628759',
        'IVdose_3_HV': '#70b5aa',
        'IVdose_10_HV': '#35978f',
        'IVdose_20_SLE': '#76b56e',
        'SCdose_50_HV': '#6d65bf'
    }

    for experiment in PK_data:
        plt.figure(figsize=(8, 5))
        timepoints = time_vectors[experiment]
        base_color = mcolors.to_rgb(dose_colors[experiment])
        dark_color = adjust_color_brightness(base_color, 0.6)
        light_color = adjust_color_brightness(base_color, 1.4)

        # --- Get min/max for high and low PDC uncertainty ---
        y_min_high = np.full_like(timepoints, np.inf, dtype=float)
        y_max_low = np.full_like(timepoints, -np.inf, dtype=float)

        for param_high in acceptable_param_sets[1]:  # high_pdc
            try:
                sim_sets[1][experiment].simulate(time_vector=timepoints, parameter_values=param_high, reset=True)
                y_sim = sim_sets[1][experiment].feature_data[:, 0]
                y_min_high = np.minimum(y_min_high, y_sim)
            except RuntimeError:
                continue

        for param_low in acceptable_param_sets[2]:  # low_pdc
            try:
                sim_sets[2][experiment].simulate(time_vector=timepoints, parameter_values=param_low, reset=True)
                y_sim = sim_sets[2][experiment].feature_data[:, 0]
                y_max_low = np.maximum(y_max_low, y_sim)
            except RuntimeError:
                continue

        # Plot gradient fill between high and low PDC uncertainty
        plot_gradient_fill_between(plt.gca(), timepoints, y_min_high, y_max_low, base_color)




        # Plot each model simulation with selected param set (same color)
        for i, label in enumerate(['Original', 'High PDC', 'Low PDC']):
            sim = sim_sets[i][experiment]
            params = selected_param_sets[i]
            try:
                sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_main = sim.feature_data[:, 0]
                plt.plot(timepoints, y_main, label=label, color=base_color, linestyle=['-', '--', ':'][i])
            except RuntimeError:
                print(f"Simulation failed for {label} on {experiment}")

        plt.xlabel('Time [Hours]')
        plt.ylabel('BIIB059 Skin Concentration (µg/ml)')
        plt.title(experiment)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{experiment}_PK_skin_gradient.png")
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.close()



plot_all_doses_with_high_low_pdc_uncertainty_gradient(
    selected_param_sets=[original_params, high_params, low_params],
    acceptable_param_sets=[acceptable_params, acceptable_params_high, acceptable_params_low],
    sim_sets=[model_sims, model_high_pdc_sims, model_low_pdc_sims],
    PK_data=PK_data,
    time_vectors=time_vectors
)


