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

# Open the mPBP_SLE_model.txt file and read its contents
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../Data/SLE_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Define a function to plot one PK_dataset
def plot_PK_dataset(PK_data, face_color='k'):
    plt.errorbar(PK_data['time'], PK_data['BIIB059_mean'], PK_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BIIB059 Concentration (µg/ml)')

# Definition of the function that plots all PK_datasets
def plot_PK_data(PK_data, face_color='k'):
    for experiment in PK_data:
        plt.figure()
        plot_PK_dataset(PK_data[experiment], face_color=face_color)
        plt.title(experiment)

# Install and load model
sund.install_model('../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())
first_model = sund.load_model("mPBPK_SLE_model")

# Creating activities for the different doses
bodyweight = 70
IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_20_SLE']['input']['IV_in']['f']))

first_model_sims = {
    'IVdose_20_SLE': sund.Simulation(models=first_model, activities=IV_20_SLE, time_unit='h'),
}

time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 1000, 1) for exp in PK_data}

def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PK_sim'):
    sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:, feature_idx], color)

def plot_sim_with_PK_data(params, sims, PK_data, color='b'):
    for experiment in PK_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_sim(params, sims[experiment], timepoints, color)
        plot_PK_dataset(PK_data[experiment])
        plt.title(experiment)

def fcost(params, sims, PK_data):
    cost = 0
    for dose in PK_data:
        try:
            sims[dose].simulate(time_vector=PK_data[dose]["time"], parameter_values=params, reset=True)
            PK_sim = sims[dose].feature_data[:,0]
            y = PK_data[dose]["BIIB059_mean"]
            SEM = PK_data[dose]["SEM"]
            cost += np.sum(np.square(((PK_sim - y) / SEM)))
        except Exception as e:
            if "CVODE" not in str(e):
                print(str(e))
            return 1e30
    return cost

params_M1 = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.77, 0.95, 0.605, 0.2, 11.095, 14.15, 0.28, 2.12e-05, 2.5, 0.525, 1.27e-5]
# Linear clearance have been updated for SLE, otherwise the same optimized parameters from HV is used

cost_M1 = fcost(params_M1, first_model_sims, PK_data)
print(f"Cost of the M1 model: {cost_M1}")

dgf = sum(np.count_nonzero(np.isfinite(PK_data[exp]["SEM"])) for exp in PK_data)
chi2_limit = chi2.ppf(0.95, dgf)
print(f"Chi2 limit: {chi2_limit}")
print(f"Cost > limit (rejected?): {cost_M1 > chi2_limit}")

plot_sim_with_PK_data(params_M1, first_model_sims, PK_data)

plt.show()

args_M1 = (first_model_sims, PK_data)
params_M1_log = np.log(params_M1)

# Bounds for the parameters
bound_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1] 
# Frozen parameters except linear clearance 

lower_bounds = np.log(params_M1) - np.log(bound_factors)
upper_bounds = np.log(params_M1) + np.log(bound_factors)
bounds_M1_log = Bounds(lower_bounds, upper_bounds)

# Load previous best result
if os.path.exists('best_SLE_PK_result.json'):
    with open('best_SLE_PK_result.json', 'r') as f:
        best_data = json.load(f)
        best_cost_PK = best_data['best_cost']
        acceptable_params_PK = [np.array(best_data['best_param'])]
else:
    best_cost_PK = np.inf
    acceptable_params_PK = []

# Cost function for optimization
def fcost_uncertainty_M1(param_log, model, PK_data):
    global acceptable_params_PK
    global best_cost_PK

    params = np.exp(param_log)
    cost = fcost(params, model, PK_data)

    if cost < best_cost_PK:
        acceptable_params_PK = [params]
        best_cost_PK = cost
        print(f"New best cost: {best_cost_PK}")

    return cost

def callback(x, file_name):
    with open(f"./{file_name}.json", 'w') as file:
        out = {"x": x}
        json.dump(out, file, cls=NumpyArrayEncoder)

def callback_log(x, file_name='PK-temp'):
    callback(np.exp(x), file_name=file_name)

def callback_PK_evolution_log(x, convergence):
    callback_log(x, file_name='PK-temp-evolution')

for i in range(5):
    res = differential_evolution(func=fcost_uncertainty_M1, bounds=bounds_M1_log, args=args_M1,
                                  x0=params_M1_log, callback=callback_PK_evolution_log, disp=True)
    params_M1_log = res['x']

with open('best_SLE_PK_result.json', 'w') as f:
    json.dump({'best_cost': best_cost_PK, 'best_param': acceptable_params_PK[0]}, f, cls=NumpyArrayEncoder)

with open('acceptable_params_SLE_PK.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(acceptable_params_PK)

def plot_uncertainty(all_params, sims, PK_data, color='b', n_params_to_plot=500):
    random.shuffle(all_params)
    for experiment in PK_data:
        print(f"\nPlotting uncertainty for: {experiment}")
        plt.figure()
        timepoints = time_vectors[experiment]
        success_count = 0
        fail_count = 0
        for param in all_params:
            if success_count >= n_params_to_plot:
                break
            try:
                plot_sim(param, sims[experiment], timepoints, color)
                success_count += 1
            except RuntimeError as e:
                if "CVODE" in str(e):
                    fail_count += 1
                    continue
                else:
                    raise e
        plot_PK_dataset(PK_data[experiment])
        plt.title(experiment)
        print(f"  Successful simulations: {success_count}")
        print(f"  Failed (CVODE) simulations: {fail_count}")

p_opt_M1 = res['x']
plot_uncertainty(acceptable_params_PK, first_model_sims, PK_data)
print(f"Chi2 limit: {chi2_limit}")
plt.show()