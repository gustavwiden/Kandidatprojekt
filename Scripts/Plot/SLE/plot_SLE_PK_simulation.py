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
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../../Data/SLE_PK_data.json", "r") as f:
    SLE_PK_data = json.load(f)

def plot_PK_dataset(PK_data, face_color='k'):
    plt.errorbar(PK_data['time'], PK_data['BIIB059_mean'], PK_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BIIB059 serum conc. (µg/ml)')

# Definition of a function that plots the simulation
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PK_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Definition of the function that plots all PK simulations and saves them to Results folder
def plot_sim_with_SLE_PK_data(params, sims, SLE_PK_data, color='b', save_dir='../../../Results/SLE_results/PK'):
    os.makedirs(save_dir, exist_ok=True)

    for experiment in SLE_PK_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_sim(params, sims[experiment], timepoints, color)
        plot_PK_dataset(SLE_PK_data[experiment], face_color=color)
        
        # Replace "_HV" with "_SLE" in the experiment name
        experiment_sle = experiment.replace("_HV", "_SLE")
        
        # Update the title
        plt.title(experiment_sle)

        # Save figure with PK-specific name
        filename = f"SLE_PK_{experiment_sle}_simulation.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

## Setup of the model

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model object
first_model = sund.load_model("mPBPK_SLE_model")

# Creating activities for the different doses
bodyweight = 70 # Bodyweight for subject in kg

IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PK_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PK_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PK_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PK_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PK_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PK_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PK_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PK_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PK_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PK_data['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PK_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PK_data['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = SLE_PK_data['SCdose_50_HV']['input']['SC_in']['t'],  f = SLE_PK_data['SCdose_50_HV']['input']['SC_in']['f'])

first_model_sims = {
    'IVdose_005_HV': sund.Simulation(models = first_model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = first_model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = first_model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = first_model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = first_model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = first_model, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = first_model, activities = SC_50_HV, time_unit = 'h')
}

time_vectors = {exp: np.arange(-10, SLE_PK_data[exp]["time"][-1] + 0.01, 1) for exp in SLE_PK_data}

params_M1 = [0.81995, 0.00867199496525978, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6927716105886019, 0.95, 0.7960584853135797, 0.2, 0.0096780180307827, 1.52, 54.7, 1.14185149185025, 14000.0]

def plot_all_PK_doses_together(params, sims, SLE_PK_data, time_vectors, save_dir='../../../Results/SLE_results/PK', feature_to_plot='PK_sim'):
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
        'IVdose_20_SLE':  'IV 20',
        'SCdose_50_HV':  'SC 50'
    }
    label_positions = {
        'IVdose_005_HV': (400, 0.05),
        'IVdose_03_HV':  (300, 0.4),
        'IVdose_1_HV':   (1480, 0.10),
        'IVdose_3_HV':   (1570, 3),
        'IVdose_10_HV':  (770, 15),
        'IVdose_20_SLE':  (1600, 30),
        'SCdose_50_HV':  (900, 0.17),
    }

    # Plotta varje simulering
    for i, (experiment, color) in enumerate(zip(SLE_PK_data.keys(), colors)):
        timepoints = time_vectors[experiment]
        sim = sims[experiment]
        sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
        feature_idx = sim.feature_names.index(feature_to_plot)

        y = sim.feature_data[:, feature_idx]
        x = sim.time_vector

        # Plot simulation curve
        plt.plot(x, y, color=color, linewidth=2, label=dose_labels.get(experiment, experiment))

        # Add datapoints for IV 20
        if experiment == 'IVdose_20_SLE':
            x_data = SLE_PK_data[experiment]['time']
            y_data = SLE_PK_data[experiment]['BIIB059_mean']
            y_err = SLE_PK_data[experiment]['SEM']
            plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color=color,
                         ecolor='#6d65bf', capsize=3, label=f'{dose_labels[experiment]} data')

        # Add manually placed labels
        if experiment in label_positions:
            label_x, label_y = label_positions[experiment]
            plt.text(label_x, label_y, dose_labels.get(experiment, experiment),
                     color=color, fontsize=18, weight='bold')

    plt.xlabel('Time [Hours]', fontsize=16)
    plt.ylabel('BIIB059 concentration (µg/ml)', fontsize=16)
    plt.title('PK simulation of all doses in SLE', fontsize=20)

    plt.yscale('log')
    plt.ylim(0.01, 700)
    plt.xlim(-25, 2150)

    plt.tight_layout()

    # Spara
    save_path = os.path.join(save_dir, "SLE_PK_all_doses_simulation.svg")
    plt.savefig(save_path, bbox_inches='tight')

    save_path = os.path.join(save_dir, "SLE_PK_all_doses_simulation.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


plot_sim_with_SLE_PK_data(params_M1, first_model_sims, SLE_PK_data)
# plot_all_PK_doses_together(params_M1, first_model_sims, SLE_PK_data, time_vectors)
