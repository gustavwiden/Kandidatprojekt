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
with open("../../../Data/PD_data.json", "r") as f:
    SLE_PD_data = json.load(f)

# Definition of a function that plots the simulation
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PD_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

def plot_PD_dataset(PD_data, face_color='k'):
    plt.errorbar(PD_data['time'], PD_data['BDCA2_median'], PD_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BDCA2 levels on pDCs (% Change from Baseline)')

# Definition of the function that plots all PD simulations and saves them to Results folder
def plot_sim_with_SLE_PD_data(params, sims, SLE_PD_data, color='b', save_dir='../../../Results/SLE_results/PD'):
    os.makedirs(save_dir, exist_ok=True)

    for experiment in SLE_PD_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_sim(params, sims[experiment], timepoints, color)
        # plot_PD_dataset(SLE_PD_data[experiment], face_color=color)
        
        # Replace "_HV" with "_SLE" in the experiment name
        experiment_sle = experiment.replace("_HV", "_SLE")
        
        # Update the title
        plt.title(experiment_sle)

        # Save figure with PD-specific name
        filename = f"SLE_PD_{experiment_sle}_simulation.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
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
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = SLE_PD_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = SLE_PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = SLE_PD_data['SCdose_50_HV']['input']['SC_in']['f'])

first_model_sims = {
    'IVdose_005_HV': sund.Simulation(models = first_model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = first_model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = first_model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = first_model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = first_model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = first_model, activities = SC_50_HV, time_unit = 'h')
}

time_vectors = {exp: np.arange(-10, SLE_PD_data[exp]["time"][-1] + 3000, 1) for exp in SLE_PD_data}

params_M1 = [0.81995, 0.00867199496525978, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6927716105886019, 0.95, 0.7960584853135797, 0.2, 0.0096780180307827, 1.52, 1.14185149185025, 14000.0]

def plot_all_PD_doses_together(params, sims, SLE_PD_data, time_vectors, save_dir='../../../Results/SLE_results/PD', feature_to_plot='PD_sim'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))  # Create a single figure for all plots

    # Colors for the plots
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#6d65bf', '#d95f02']

    # Shorter labels for the doses
    dose_labels = {
        'IVdose_005_HV': 'IV 0.05',
        'IVdose_03_HV':  'IV 0.3',
        'IVdose_1_HV':   'IV 1',
        'IVdose_3_HV':   'IV 3',
        'IVdose_20_SLE':  'IV 20',
        'SCdose_50_HV':  'SC 50'
    }

    # Plot each simulation
    for i, (experiment, color) in enumerate(zip(SLE_PD_data.keys(), colors)):
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
            x_data = SLE_PD_data[experiment]['time']
            y_data = SLE_PD_data[experiment]['BDCA2_median']
            y_err = SLE_PD_data[experiment]['SEM']
            plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color=color,
                         ecolor='#6d65bf', capsize=3, label=f'{dose_labels[experiment]} data')

    # Add labels, title, and legend
    plt.xlabel('Time [Hours]', fontsize=16)
    plt.ylabel('BDCA2 levels on pDCs (% Change from Baseline)', fontsize=16)
    plt.title('PD simulation of all doses in SLE', fontsize=20)
    plt.legend()
    plt.tight_layout()
    plt.xlim(-100, 5100)
    plt.ylim(-110, 50)

    # Save the plot
    save_path = os.path.join(save_dir, "SLE_PD_all_doses_simulation.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    save_path = os.path.join(save_dir, "SLE_PD_all_doses_simulation.svg")
    plt.savefig(save_path, bbox_inches='tight')

    plt.show()  # Display the figure


plot_sim_with_SLE_PD_data(params_M1, first_model_sims, SLE_PD_data)
# plot_all_PD_doses_together(params_M1, first_model_sims, SLE_PD_data, time_vectors)
