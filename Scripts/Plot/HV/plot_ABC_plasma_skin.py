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
from scipy.stats import linregress

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params.json", "r") as f:
    acceptable_params = json.load(f)

# Install the model
sund.install_model('../../../Models/mPBPK_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_model")

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

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data['SCdose_50_HV']['input']['SC_in']['f'])

model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

HV_params = [0.5982467918487137, 0.013501146489749132, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.95, 0.95, 0.7467544604963505, 0.2, 0.00549200604682213, 0.9621937056820449, 5.539999999999999, 2623.9999999999995]

def plot_ABC_plasma_vs_skin(selected_params, sims, PK_data, time_vectors, save_dir='../../../Results/HV/PK'):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
    dose_labels = ['0.05 mg/kg', '0.3 mg/kg', '1 mg/kg', '3 mg/kg', '10 mg/kg', '20 mg/kg', '50 mg/kg SC']
    dose_keys = ['IVdose_005_HV', 'IVdose_03_HV', 'IVdose_1_HV', 'IVdose_3_HV', 'IVdose_10_HV', 'IVdose_20_HV', 'SCdose_50_HV']

    all_skin = []
    all_plasma = []
    all_colors = []

    plt.figure(figsize=(8, 6))

    for i, key in enumerate(dose_keys):
        if key not in PK_data:
            continue
        color = colors[i % len(colors)]
        label = dose_labels[i]
        timepoints = time_vectors[key]
        sims[key].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        skin = sims[key].feature_data[:, 2]
        plasma = sims[key].feature_data[:, 0]

        # Only keep points where both values are positive
        valid = (skin > 0) & (plasma > 0)
        skin = skin[valid]
        plasma = plasma[valid]

        plt.scatter(plasma, skin, color=color, label=label, alpha=0.7)
        all_skin.extend(skin)
        all_plasma.extend(plasma)
        all_colors.extend([color]*len(skin))

    # Plot reference lines
    x_min = min(all_plasma)
    x_max = max(all_plasma)
    x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)

    plt.plot(x_fit, 0.157 * x_fit, 'k-')
    plt.plot(x_fit, 0.0785 * x_fit, 'k--')
    plt.plot(x_fit, 0.314 * x_fit, 'k--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Plasma Concentration (µg/ml)')
    plt.ylabel('Skin Concentration (µg/ml)')
    plt.title('Model Simulations for Skin vs Plasma Concentration in Healthy Volunteers')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    save_path_png = os.path.join(save_dir, "Skin vs Plasma Concentration.png")
    plt.savefig(save_path_png, format='png', dpi=600)
    plt.show()


plot_ABC_plasma_vs_skin(HV_params, model_sims, PK_data, time_vectors)

