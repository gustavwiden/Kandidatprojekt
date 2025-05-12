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
with open("../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

# Definition of a function that plots the simulation
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PD_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Definition of the function that plots all PD simulations and saves them to Results folder
def plot_sim_with_PD_data(params, sims, PD_data, color='b', save_dir='../Results/SLE_results/PD'):
    os.makedirs(save_dir, exist_ok=True)

    for experiment in PD_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_sim(params, sims[experiment], timepoints, color)
        
        # Replace "_HV" with "_SLE" in the experiment name
        experiment_sle = experiment.replace("_HV", "_SLE")
        
        # Update the title
        plt.title(experiment_sle)

        # Save figure with PD-specific name
        filename = f"SLE_PD_{experiment_sle}_simulation.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

## Setup of the model

# Install the model
sund.install_model('../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model object
first_model = sund.load_model("mPBPK_SLE_model")

# Creating activities for the different doses
bodyweight = 70 # Bodyweight for subject in kg

IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PD_data['SCdose_50_HV']['input']['SC_in']['f'])

first_model_sims = {
    'IVdose_005_HV': sund.Simulation(models = first_model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = first_model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = first_model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = first_model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = first_model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = first_model, activities = SC_50_HV, time_unit = 'h')
}

time_vectors = {exp: np.arange(-10, PD_data[exp]["time"][-1] + 0.01, 1) for exp in PD_data}

params_M1 = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.769, 0.95, 0.605, 
0.2, 10.43, 20900, 281, 1.31e-1, 8, 525, 0.07]

def plot_all_PD_doses_together(params, sims, PD_data, time_vectors, save_dir='../Results/SLE_results/PD', feature_to_plot='PD_sim'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))

    # Färger (välj fler eller andra om du vill)
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#76b56e', '#6d65bf', '#d95f02']

    # Kortare etiketter
    dose_labels = {
        'IVdose_005_HV': 'IV 0.05',
        'IVdose_03_HV':  'IV 0.3',
        'IVdose_1_HV':   'IV 1',
        'IVdose_3_HV':   'IV 3',
        'IVdose_10_HV':  'IV 10',
        'IVdose_20_HV':  'IV 20',
        'SCdose_50_HV':  'SC 50'
    }

    # Plotta varje simulering
    for i, (experiment, color) in enumerate(zip(PD_data.keys(), colors)):
        timepoints = time_vectors[experiment]
        sim = sims[experiment]
        sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
        feature_idx = sim.feature_names.index(feature_to_plot)

        y = sim.feature_data[:, feature_idx]
        x = sim.time_vector

        plt.plot(x, y, color=color, linewidth=2, label=dose_labels.get(experiment, experiment))

    plt.xlabel('Time [Hours]', fontsize=16)
    plt.ylabel('BIIB059 concentration (µg/ml)', fontsize=16)
    plt.title('PD simulation of all doses in SLE', fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Spara
    save_path = os.path.join(save_dir, "SLE_PD_all_doses_simulation.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


plot_sim_with_PD_data(params_M1, first_model_sims, PD_data)
plot_all_PD_doses_together(params_M1, first_model_sims, PD_data, time_vectors)
