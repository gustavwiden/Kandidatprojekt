import os
import re
import numpy as np
import math as m
import matplotlib.pyplot as plt
import sund
import json
from scipy.stats import chi2
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
from scipy.integrate import odeint
import sys
import csv
import random
import requests
import pypesto
import pypesto.optimize as optimize

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the PK data file and read its contents
with open("../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

# Open the PD data file and read its contents
with open("../../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

# Install and load the model
sund.install_model('../../Models/mPBPK_model.txt')
print(sund.installed_models())
model = sund.load_model("mPBPK_model")

# Bodyweight for subject in kg
bodyweight = 73

# Create activity objects for each dose
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

# Create simulation objects for each dose
model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

# Create time vectors for each experiment
# The time vectors are created based on the maximum time in the PK_data and PD_data for each experiment
time_vectors_PK = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}
time_vectors_PD = {exp: np.arange(-10, PD_data[exp]["time"][-1] + 0.01, 1) for exp in PD_data}

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
initial_params = [0.713, 0.00975, 2.6, 1.125, 6.987, 4.368, 2.6, 0.0065, 0.0338, 0.081, 0.63, 0.95, 0.4, 0.2, 0.00552, 10.55, 0.83, 16800, 1e9]

# Print cost for initial parameters
cost = fcost_joint(initial_params, model_sims, PK_data, PD_data)
print(f"Joint cost: {cost[0]:.2f}, PK cost: {cost[1]:.2f}, PD cost: {cost[2]:.2f}")

# Calculate the degrees of freedom and chi2-limits for PK and PD separately
dgf_PK = sum(np.count_nonzero(np.isfinite(PK_data[exp]["SEM"])) for exp in PK_data)
dgf_PD = sum(np.count_nonzero(np.isfinite(PD_data[exp]["SEM"])) for exp in PD_data)
chi2_limit_PK = chi2.ppf(0.95, dgf_PK)
chi2_limit_PD = chi2.ppf(0.95, dgf_PD)

# Print the chi-squared limit and whether the cost exceeds it
print(f"PK Chi2 limit: {chi2_limit_PK}", f"PD Chi2 limit: {chi2_limit_PD}")

# Define a callback function to save the optimization results
# This function saves the current parameters and their cost to a JSON file
def callback(x, file_name):
    with open(f"./{file_name}.json",'w') as file:
        out = {"x": x}
        json.dump(out,file, cls=NumpyArrayEncoder)

# Define the cost function arguments
# This is a tuple containing the model simulations and PK data
cost_function_args = (model_sims, PK_data, PD_data)

# Convert the initial parameters to logarithmic scale for optimization
initial_params_log = np.log(initial_params)

# Bounds for the parameters
bound_factors = [1.2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1.1, 1, 2, 1, 2, 1, 10, 1.2, 1000]

# Calculate the logarithmic bounds for the parameters
# The bounds are defined as log(initial_params) ± log(bound_factors)
lower_bounds = np.log(initial_params) - np.log(bound_factors)
upper_bounds = np.log(initial_params) + np.log(bound_factors)
bounds_log = Bounds(lower_bounds, upper_bounds)

# Print the bounds for the parameters
print("Lower bounds:", np.exp(lower_bounds))
print("Upper bounds:", np.exp(upper_bounds))

# Define the cost function for the optimization in logarithmic scale
# This function takes parameters in logarithmic scale, exponentiates them, and then calls the original cost function
def fcost_log(params_log, sims, PK_data, PD_data):
    return fcost(np.exp(params_log.copy()), sims, PK_data, PD_data)     

# Define a callback function for logging the optimization progress
# This function converts the parameter to original scale and calls the callback function to save the parameters
def callback_log(x, file_name='PK-temp'):
    callback(np.exp(x), file_name=file_name)

# Define a callback function for logging the evolution of the optimization
# This function is called at each iteration of the optimization and calls the callback_log_PK function
def callback_evolution_log(x,convergence):
    callback_log(x, file_name='PK-temp-evolution')

output_dir = '../../Results/Acceptable params'
os.makedirs(output_dir, exist_ok=True)
best_result_path = os.path.join(output_dir, 'best_result.json')

# Load previous best result if available
if os.path.exists(best_result_path) and os.path.getsize(best_result_path) > 0:
    with open(best_result_path, 'r') as f:
        best_data = json.load(f)
        best_cost = best_data['best_cost']
        best_param = np.array(best_data['best_param'])
else:
    best_cost = np.inf
    best_param = None

# Create list to store all acceptable parameter sets
acceptable_params = []

# Load existing acceptable parameter sets if the file exists
acceptable_params_path = os.path.join(output_dir, 'acceptable_params.json')
if os.path.exists(acceptable_params_path) and os.path.getsize(acceptable_params_path) > 0:
    with open(acceptable_params_path, 'r') as f:
        acceptable_params = json.load(f)

def fcost_optimization(param_log, model_sims, PK_data, PD_data):
    global acceptable_params
    global best_cost
    global best_param

    params = np.exp(param_log)
    joint_cost, pk_cost, pd_cost = fcost_joint(params, model_sims, PK_data, PD_data)

    # Only accept parameter sets that are below BOTH chi2 limits
    if pk_cost < chi2_limit_PK and pd_cost < chi2_limit_PD:
        acceptable_params.append(params)

    if joint_cost < best_cost:
        best_cost = joint_cost
        best_param = params.copy()
        print(f"New best joint cost: {best_cost} (PK: {pk_cost}, PD: {pd_cost})")

    return joint_cost


# for i in range(1):  # Run the optimization 1 time
#     res = differential_evolution(
#         func=fcost_optimization,
#         bounds=bounds_log,
#         args = cost_function_args,
#         x0=initial_params_log,
#         callback=callback_evolution_log,
#         disp=True
#     )


with open(os.path.join(output_dir, 'sampling_result.json'), 'r') as f:
    result_sampling = json.load(f)

# Indices of the parameters that were optimized
selected_indices = [0, 1, 10, 12, 14, 16, 17, 18]  # Must match your optimization setup

# Load best_param to fill in non-optimized values
with open(os.path.join(output_dir, 'best_result.json'), 'r') as f:
    best_data = json.load(f)
    best_param = np.array(best_data['best_param'])


trace_array = np.array(result_sampling["trace_x"][0])  # shape: (n_samples, n_params)
trace_array = trace_array[:50000]

# === Filtrera trace enligt chi2-gränser för PK och PD ===
filtered_trace = []

filtered_trace = []

for param_reduced in trace_array:
    # Create full param set by inserting reduced set into best_param
    full_param = best_param.copy()
    full_param[selected_indices] = param_reduced
    _, pk_cost, pd_cost = fcost_joint(full_param, model_sims, PK_data, PD_data)
    
    if pk_cost < chi2_limit_PK and pd_cost < chi2_limit_PD:
        filtered_trace.append(param_reduced)  # keep only reduced set for plotting

filtered_trace = np.array(filtered_trace)


print(f"Totalt antal samples: {len(trace_array)}")
print(f"Godkända enligt chi2-gräns: {len(filtered_trace)}")

# === Plotta histogram för de första 8 parametrarna ===

parameter_names = ['F', 'ka', 'RCS', 'RC2', 'CL', 'kint', 'kd', 'pdc_count_lymph']
rows, cols = 2, 4
fig, axs = plt.subplots(rows, cols, figsize=(12, 6))
axes = axs.flatten()

for i, ax in enumerate(axes):
    if i < filtered_trace.shape[1]:
        ax.hist(filtered_trace[:, i], bins='auto', edgecolor='black')
        ax.set_title(parameter_names[i])
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Frequency')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
