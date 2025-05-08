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
with open("../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

# Define a function to plot one PK_dataset
def plot_PK_dataset(PK_data, face_color='k'):
    plt.errorbar(PK_data['time'], PK_data['BIIB059_mean'], PK_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BIIB059 serum conc. (µg/ml)')

# Definition of the function that plots all PK_datasets
def plot_PK_data(PK_data, face_color='k'):
    for experiment in PK_data:
        plt.figure()
        plot_PK_dataset(PK_data[experiment], face_color=face_color)
        plt.title(experiment)

## Setup of the model

# Install the model
sund.install_model('../Models/mPBPK_model.txt')
print(sund.installed_models())

# Load the model object
first_model = sund.load_model("mPBPK_model")

# Creating activities for the different doses
bodyweight = 70 # Bodyweight for subject in kg

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

first_model_sims = {
    'IVdose_005_HV': sund.Simulation(models = first_model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = first_model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = first_model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = first_model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = first_model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = first_model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = first_model, activities = SC_50_HV, time_unit = 'h')
}

time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PK_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

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
            sims[dose].simulate(time_vector = PK_data[dose]["time"], parameter_values = params, reset = True)
            PK_sim = sims[dose].feature_data[:,0]
            y = PK_data[dose]["BIIB059_mean"]
            SEM = PK_data[dose]["SEM"] 
            cost += np.sum(np.square(((PK_sim - y) / SEM)))
        except Exception as e:
            if "CVODE" not in str(e):
                print(str(e))
            return 1e30
    return cost

params_M1 = [0.713, 0.00975, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.7, 0.95, 0.55, 
0.20, 5.52, 10.7, 0.547, 1.31e-4, 2.5e-8, 4, 4, 0.35] # Initial guess for PK and PD parameters

cost_M1 = fcost(params_M1, first_model_sims, PK_data)
print(f"Cost of the M1 model: {cost_M1}")

dgf = sum(np.count_nonzero(np.isfinite(PK_data[exp]["SEM"])) for exp in PK_data)
chi2_limit = chi2.ppf(0.95, dgf)
print(f"Chi2 limit: {chi2_limit}")
print(f"Cost > limit (rejected?): {cost_M1 > chi2_limit}")

plot_sim_with_PK_data(params_M1, first_model_sims, PK_data)

#plt.show()

def callback(x, file_name):
    with open(f"./{file_name}.json",'w') as file:
        out = {"x": x}
        json.dump(out,file, cls=NumpyArrayEncoder)

args_M1 = (first_model_sims, PK_data)
params_M1_log = np.log(params_M1)

# Bounds for the parameters
bound_factors = [1.05, 1.05, 1, 1, 1, 1, 1, 1, 1, 1, 1.1, 1, 1.1, 1, 2, 1.3, 1.3, 1, 1, 1, 1, 1] # PD parameters frozen

lower_bounds = np.log(params_M1) - np.log(bound_factors)
upper_bounds = np.log(params_M1) + np.log(bound_factors)
bounds_M1_log = Bounds(lower_bounds, upper_bounds)

print("Lower bounds:", np.exp(lower_bounds))
print("Upper bounds:", np.exp(upper_bounds))

# Cost function for the optimization
def fcost_log(params_log, sims, PK_data):
    return fcost(np.exp(params_log.copy()), sims, PK_data)     

def callback_log(x, file_name='M1-temp'):
    callback(np.exp(x), file_name=file_name)

def callback_M1_evolution_log(x,convergence):
    callback_log(x, file_name='M1-temp-evolution')

# Load previous best if exists
if os.path.exists('best_M1_result.json'):
    with open('best_M1_result.json', 'r') as f:
        best_data = json.load(f)
        best_cost_M1 = best_data['best_cost']
        acceptable_params_PK = [np.array(best_data['best_param'])]
else:
    best_cost_M1 = np.inf
    acceptable_params_PK = []

def fcost_uncertainty_M1(param_log, model, PK_data):
    global acceptable_params_PK
    global best_cost_M1

    params = np.exp(param_log)
    cost = fcost(params, model, PK_data)

    if cost < best_cost_M1:
        acceptable_params_PK = [params]
        best_cost_M1 = cost
        print(f"New best cost: {best_cost_M1}")

    return cost

for i in range(0,5):
    res = differential_evolution(func=fcost_uncertainty_M1, bounds=bounds_M1_log, args=args_M1, x0=params_M1_log, callback=callback_M1_evolution_log, disp=True)
    params_M1_log = res['x']

print(f"Number of parameter sets collected: {len(acceptable_params_PK)}")

# Save best parameter set to JSON
with open('best_M1_result.json', 'w') as f:
    json.dump({'best_cost': best_cost_M1, 'best_param': acceptable_params_PK[0]}, f, cls=NumpyArrayEncoder)

with open('acceptable_params_PK.csv', 'w', newline='') as csvfile:
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
