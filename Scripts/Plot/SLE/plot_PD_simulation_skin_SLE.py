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

sund.install_model('../../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Open the data file and read its contents
with open("../../../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

# Load the model object
mPBPK_model = sund.load_model("mPBPK_SLE_model")

time_vectors = {exp: np.arange(-10, PD_data[exp]["time"][-1] + 4000, 1) for exp in PD_data}


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

mPBPK_model_sims = {
    'IVdose_005_HV': sund.Simulation(models = mPBPK_model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = mPBPK_model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = mPBPK_model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = mPBPK_model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = mPBPK_model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = mPBPK_model, activities = SC_50_HV, time_unit = 'h')
}

# Definition of a function that plots the simulation
def plot_sim(params, sim, timepoints, color='g', feature_to_plot='Skin_PD_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Definition of the function that plots all PD simulations and saves them to Results folder
def plot_fig(params, sims, color='g', save_dir='../Results/Skin_SLE/PD'):
    os.makedirs(save_dir, exist_ok=True)

    for experiment in PD_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_sim(params, sims[experiment], timepoints, color)
        plt.title(f"PD simulation in SLE skin for {experiment}")
        plt.xlabel('Time [Hours]')
        plt.ylabel('BDCA2 levels on pDCs (% Change from Baseline)')

        # Save figure with PD-specific name
        filename = f"PD_{experiment}_simulation_in_SLE_skin.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')

params_SLE = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.77, 0.95,
            0.605, 0.2, 8.91, 14.15, 0.28, 2.12e-05, 2.5, 0.525, 1.27e-5]

plot_fig(params_SLE, mPBPK_model_sims)