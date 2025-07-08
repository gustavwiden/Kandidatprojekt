# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sund
import json
from scipy.stats import chi2
from scipy.optimize import Bounds, differential_evolution
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

# Open the mPBPK_SLE_model.txt file and read its contents
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../Data/SLE_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Open the data file and read its contents
with open("../../Data/SLE_PD_data.json", "r") as f:
    PD_data = json.load(f)

# Define a function to plot one PK_dataset
def plot_PK_dataset(PK_data, face_color='k'):
    plt.errorbar(PK_data['time'], PK_data['BIIB059_mean'], PK_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BIIB059 serum conc. (µg/ml)')

# Define a function to plot one PD dataset
def plot_PD_dataset(PD_data, face_color='k'):
    plt.errorbar(PD_data['time'], PD_data['BDCA2_median'], PD_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BDCA2 levels on pDCs, percentage change from baseline')

# Definition of the function that plots all PK_datasets
def plot_PK_data(PK_data, face_color='k'):
    for experiment in PK_data:
        plt.figure()
        plot_PK_dataset(PK_data[experiment], face_color=face_color)
        plt.title(experiment)

# Define a function to plot all PD datasets
def plot_PD_data(PD_data, face_color='k'):
    for experiment in PD_data:
        plt.figure()
        plot_PD_dataset(PD_data[experiment], face_color=face_color)
        plt.title(experiment)

# Install and load the model
sund.install_model('../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())
model = sund.load_model("mPBPK_SLE_model")

# Bodyweight for subject in kg
bodyweight = 69

# Create activity objects for each dose
IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_20_SLE']['input']['IV_in']['f']))

# Create simulation objects for each dose
model_sims = {
    'IVdose_20_SLE': sund.Simulation(models = model, activities = IV_20_SLE, time_unit = 'h')
}

# Create time vectors for each experiment
time_vectors_PK = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}
time_vectors_PD = {exp: np.arange(-10, PD_data[exp]["time"][-1] + 0.01, 1) for exp in PD_data}


# Define a function to plot the simulation results
def plot_sim_PK(params, sim, timepoints, color='b', feature_to_plot='PK_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Define a function to plot the simulation results with PK data
def plot_sim_with_PK_data(params, sims, PK_data, color='b'):
    for experiment in PK_data:
        plt.figure()
        timepoints = time_vectors_PK[experiment]
        plot_PK_dataset(PK_data[experiment])
        plot_sim_PK(params, sims[experiment], timepoints, color)
        plt.title(experiment)

def plot_sim_PD(params, sim, timepoints, color='b', feature_to_plot='PD_sim'):
    sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:, feature_idx], color)

# Define a function to plot the simulation results with PD data
def plot_sim_with_PD_data(params, sims, PD_data, color='b'):
    for experiment in PD_data:
        plt.figure()
        timepoints = time_vectors_PD[experiment]
        plot_sim_PD(params, sims[experiment], timepoints, color)
        plot_PD_dataset(PD_data[experiment])
        plt.title(experiment)

# Define the joint cost function for the optimization
# This function calculates the cost based on the difference between simulations and PK/PD data
def fcost_joint(params, sims, PK_data, PD_data, pk_weight=1.0, pd_weight=1.0):
    # PK cost
    pk_cost = 0
    for dose in PK_data:
        try:
            sims[dose].simulate(time_vector=PK_data[dose]["time"], parameter_values=params, reset=True)
            PK_sim = sims[dose].feature_data[:, sims[dose].feature_names.index('PK_sim')]
            y = PK_data[dose]["BIIB059_mean"]
            SEM = PK_data[dose]["SEM"]
            pk_cost += np.sum(np.square(((PK_sim - y) / SEM)))
        except Exception as e:
            if "CVODE" not in str(e):
                print(str(e))
            return 1e30, 1e30, 1e30  # Return large costs if simulation fails

    # PD cost
    pd_cost = 0
    for dose in PD_data:
        try:
            sims[dose].simulate(time_vector=PD_data[dose]["time"], parameter_values=params, reset=True)
            PD_sim = sims[dose].feature_data[:, sims[dose].feature_names.index('PD_sim')]
            y = PD_data[dose]["BDCA2_median"]
            SEM = PD_data[dose]["SEM"]
            pd_cost += np.sum(np.square(((PD_sim - y) / SEM)))
        except Exception as e:
            if "CVODE" not in str(e):
                print(str(e))
            return 1e30, 1e30, 1e30

    joint_cost = pk_weight * pk_cost + pd_weight * pd_cost
    return joint_cost, pk_cost, pd_cost

# Define the initial guesses for the parameters
initial_params = [0.70167507023512, 0.010970491553609206, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.5908548614616957, 0.95, 0.7272247648651022, 0.2, 0.00552, 7.23, 66.97, 0.83, 14123.510378662331, 80063718.67276345]

# Print cost for initial parameters
cost = fcost_joint(initial_params, model_sims, PK_data, PD_data)
print(f"Joint cost: {cost[0]:.2f}, PK cost: {cost[1]:.2f}, PD cost: {cost[2]:.2f}")

# Calculate the degrees of freedom and chi2-limits for PK and PD separately
dgf_PK = sum(np.count_nonzero(np.isfinite(PK_data[exp]["SEM"])) for exp in PK_data)
dgf_PD = sum(np.count_nonzero(np.isfinite(PD_data[exp]["SEM"])) for exp in PD_data)
chi2_limit_PK = chi2.ppf(0.95, dgf_PK)
chi2_limit_PD = chi2.ppf(0.95, dgf_PD)

print(f"PK Chi2 limit: {chi2_limit_PK:.2f}", f"PD Chi2 limit: {chi2_limit_PD:.2f}")

# Plot the simulation results with PK and PD data
plot_sim_with_PK_data(initial_params, model_sims, PK_data)
plot_sim_with_PD_data(initial_params, model_sims, PD_data)
plt.show()

# Callback and optimization setup
def callback(x, file_name):
    with open(f"./{file_name}.json",'w') as file:
        out = {"x": x}
        json.dump(out,file, cls=NumpyArrayEncoder)

cost_function_args = (model_sims, PK_data, PD_data)
initial_params_log = np.log(initial_params)
bound_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 10, 1, 1]
lower_bounds = np.log(initial_params) - np.log(bound_factors)
upper_bounds = np.log(initial_params) + np.log(bound_factors)
bounds_log = Bounds(lower_bounds, upper_bounds)
print("Lower bounds:", np.exp(lower_bounds))
print("Upper bounds:", np.exp(upper_bounds))

def callback_log(x, file_name='SLE-temp'):
    callback(np.exp(x), file_name=file_name)

def callback_evolution_log(x,convergence):
    callback_log(x, file_name='SLE-temp-evolution')

output_dir = '../../Results/Acceptable params'
os.makedirs(output_dir, exist_ok=True)
best_result_path = os.path.join(output_dir, 'best_SLE_result.json')

if os.path.exists(best_result_path) and os.path.getsize(best_result_path) > 0:
    with open(best_result_path, 'r') as f:
        best_data = json.load(f)
        best_cost = best_data['best_cost']
        best_param = np.array(best_data['best_param'])
else:
    best_cost = np.inf
    best_param = None

acceptable_params = []
acceptable_params_path = os.path.join(output_dir, 'acceptable_params_SLE.json')
if os.path.exists(acceptable_params_path) and os.path.getsize(acceptable_params_path) > 0:
    with open(acceptable_params_path, 'r') as f:
        acceptable_params = json.load(f)

# Cost function for uncertainty analysis (accept only if both PK and PD are below their chi2 limits)
def fcost_uncertainty(param_log, model, PK_data, PD_data):
    global acceptable_params
    global best_cost
    global best_param

    params = np.exp(param_log)
    joint_cost, pk_cost, pd_cost = fcost_joint(params, model, PK_data, PD_data)

    # Only accept parameter sets that are below BOTH chi2 limits
    if pk_cost < chi2_limit_PK and pd_cost < chi2_limit_PD:
        acceptable_params.append(params)

    if joint_cost < best_cost:
        best_cost = joint_cost
        best_param = params.copy()
        print(f"New best joint cost: {best_cost:.2f} (PK: {pk_cost:.2f}, PD: {pd_cost:.2f})")

    return joint_cost

# Perform optimization using differential evolution
for i in range(5):  # Run the optimization 5 times
    res = differential_evolution(
        func=fcost_uncertainty,
        bounds=bounds_log,
        args=cost_function_args,
        x0=initial_params_log,
        callback=callback_evolution_log,
        disp=True
    )

# Save all acceptable parameter sets to a CSV file
with open(os.path.join(output_dir, 'acceptable_params_SLE.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(acceptable_params)

# Save all acceptable parameter sets to a JSON file
with open(os.path.join(output_dir, 'acceptable_params_SLE.json'), 'w') as f:
    json.dump(acceptable_params, f, cls=NumpyArrayEncoder)

# Save the best parameter set to a JSON file
with open(os.path.join(output_dir, 'best_SLE_result.json'), 'w') as f:
    json.dump({'best_cost': best_cost, 'best_param': best_param.tolist()}, f, cls=NumpyArrayEncoder)

print(f"Number of acceptable parameter sets collected: {len(acceptable_params)}")

# Plot the uncertainty of PK in the model using the acceptable parameters
def plot_uncertainty(all_params, sims, PK_data, PD_data, color='b', n_params_to_plot=500):
    random.shuffle(all_params)

    for experiment in PK_data:
        print(f"\nPlotting uncertainty for: {experiment}")
        plt.figure()
        timepoints = time_vectors_PK[experiment]
        success_count = 0
        fail_count = 0
        for param in all_params:
            if success_count >= n_params_to_plot:
                break
            try:
                plot_sim_PK(param, sims[experiment], timepoints, color)
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

    for experiment in PD_data:
        print(f"\nPlotting uncertainty for: {experiment}")
        plt.figure()
        timepoints = time_vectors_PD[experiment]
        success_count = 0
        fail_count = 0
        for param in all_params:
            if success_count >= n_params_to_plot:
                break
            try:
                plot_sim_PD(param, sims[experiment], timepoints, color)
                success_count += 1
            except RuntimeError as e:
                if "CVODE" in str(e):
                    fail_count += 1
                    continue
                else:
                    raise e
        plot_PD_dataset(PD_data[experiment])
        plt.title(experiment)
        print(f"  Successful simulations: {success_count}")
        print(f"  Failed (CVODE) simulations: {fail_count}")

plot_uncertainty(acceptable_params, model_sims, PK_data, PD_data)

# Show the plots
plt.show()
