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
with open("../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

# Define a function to plot one PD_dataset
def plot_PD_dataset(PD_data, face_color='k'):
    plt.errorbar(PD_data['time'], PD_data['BDCA2_median'], PD_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BDCA2 levels on pDCs, percentage change from baseline. (µg/ml)')

# Definition of a function that plots the simulation
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PD_sim'):
    sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:, feature_idx], color)

# Definition of the function that plots all PD simulations and saves them to Results folder
def plot_sim_with_PD_data(params, sims, PD_data, color='b', save_dir='../Results'):
    os.makedirs(save_dir, exist_ok=True)

    for experiment in PD_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_sim(params, sims[experiment], timepoints, color)
        plot_PD_dataset(PD_data[experiment])
        plt.title(experiment)

        # Save figure with PD-specific name
        filename = f"PD_{experiment}_simulation.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# Install and load model
sund.install_model('../Models/mPBPK_model.txt')
print(sund.installed_models())
first_model = sund.load_model("mPBPK_model")

# Creating activities for the different doses
bodyweight = 70
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
    'IVdose_005_HV': sund.Simulation(models=first_model, activities=IV_005_HV, time_unit='h'),
    'IVdose_03_HV': sund.Simulation(models=first_model, activities=IV_03_HV, time_unit='h'),
    'IVdose_1_HV': sund.Simulation(models=first_model, activities=IV_1_HV, time_unit='h'),
    'IVdose_3_HV': sund.Simulation(models=first_model, activities=IV_3_HV, time_unit='h'),
    'IVdose_20_HV': sund.Simulation(models=first_model, activities=IV_20_HV, time_unit='h'),
    'SCdose_50_HV': sund.Simulation(models=first_model, activities=SC_50_HV, time_unit='h')
}

time_vectors = {exp: np.arange(-10, PD_data[exp]["time"][-1] + 1000, 1) for exp in PD_data}

def fcost(params, sims, PD_data):
    cost = 0
    for dose in PD_data:
        try:
            sims[dose].simulate(time_vector=PD_data[dose]["time"], parameter_values=params, reset=True)
            PD_sim = sims[dose].feature_data[:,0]
            y = PD_data[dose]["BDCA2_median"]
            SEM = PD_data[dose]["SEM"]
            cost += np.sum(np.square(((PD_sim - y) / SEM)))
        except Exception as e:
            if "CVODE" not in str(e):
                print(str(e))
            return 1e30
    return cost

params_M1 = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.769, 0.95, 0.605, 0.2, 
5.896, 13.9, 0.421, 1.09e-4, 5e-8, 8, 8, 0.525] # Optimized parameters both models

cost_M1 = fcost(params_M1, first_model_sims, PD_data)
print(f"Cost of the M1 model: {cost_M1}")

dgf = sum(np.count_nonzero(np.isfinite(PD_data[exp]["SEM"])) for exp in PD_data)
chi2_limit = chi2.ppf(0.95, dgf)
print(f"Chi2 limit: {chi2_limit}")
print(f"Cost > limit (rejected?): {cost_M1 > chi2_limit}")

plot_sim_with_PD_data(params_M1, first_model_sims, PD_data)
