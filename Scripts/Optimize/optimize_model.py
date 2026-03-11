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
    
# Load healthy volunteers (HV) PK data
with open("../../Data/PK_data.json", "r") as f:
    HV_PK_data = json.load(f)

# Load HV PD data
with open("../../Data/PD_data.json", "r") as f:
    HV_PD_data = json.load(f)
# Load SLE PK data
with open("../../Data/SLE_PK_data.json", "r") as f:
    SLE_PK_data = json.load(f)

# Load SLE PD data
with open("../../Data/SLE_PD_data.json", "r") as f:
    SLE_PD_data = json.load(f)

# Open the mPBPK_model.txt file and read its contents
with open("../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the mPBPK_SLE_model_80_pdc_mm2.txt file and read its contents
with open("../../Models/mPBPK_SLE_model_80_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

# Install and load the models
sund.install_model('../../Models/mPBPK_model.txt')
sund.install_model('../../Models/mPBPK_SLE_model_80_pdc_mm2.txt')
print(sund.installed_models())

HV_model = sund.load_model("mPBPK_model")
SLE_model = sund.load_model("mPBPK_SLE_model_80_pdc_mm2")

HV_data = {'PK': HV_PK_data, 'PD': HV_PD_data}
SLE_data = {'PK': SLE_PK_data, 'PD': SLE_PD_data}
# SLE_data = {'PK': SLE_PK_data, 'PD': SLE_PD_data, 'PD_skin': SLE_PD_skin_data}

all_models = {'HV': HV_model, 'SLE': SLE_model}
all_datasets = {'HV': HV_data, 'SLE': SLE_data}

# Average bodyweight (kg) for HV and SLE patients (cohort 1-7 and cohort 8 respectively in the phase 1 trial)
bodyweights = {'HV': 73, 'SLE': 69}  

# Create time vectors for each experiment
time_vectors_PK = {dose: np.arange(-10, HV_PK_data[dose]["time"][-1] + 0.01, 1) for dose in HV_PK_data}
time_vectors_PD = {dose: np.arange(-10, HV_PD_data[dose]["time"][-1] + 0.01, 1) for dose in HV_PD_data}
time_vectors_SLE_PK = {dose: np.arange(-10, SLE_PK_data[dose]["time"][-1] + 0.01, 1) for dose in SLE_PK_data}
time_vectors_SLE_PD = {dose: np.arange(-10, SLE_PD_data[dose]["time"][-1] + 0.01, 1) for dose in SLE_PD_data}
# time_vectors_SLE_PD_skin = {dose: np.arange(-10, SLE_PD_skin_data[dose]["time"][-1] + 3000, 1) for dose in SLE_PD_skin_data}

HV_time_vectors = {'PK': time_vectors_PK, 'PD': time_vectors_PD}
SLE_time_vectors = {'PK': time_vectors_SLE_PK, 'PD': time_vectors_SLE_PD}
# SLE_time_vectors = {'PK': time_vectors_SLE_PK, 'PD': time_vectors_SLE_PD, 'PD_skin': time_vectors_SLE_PD_skin}

all_time_vectors = {'HV': HV_time_vectors, 'SLE': SLE_time_vectors}

measurements = {'PK': 'BIIB059_mean', 'PD': 'BDCA2_median'}
# measurements = {'PK': 'BIIB059_mean', 'PD': 'BDCA2_median', 'PD_skin': 'BDCA2_median'}
ylabels = {'PK': 'Free Litifilimab Plasma Concentration [µg/ml]', 'PD': 'Total BDCA2 Expression on pDCs [% Change]'}
# ylabels = {'PK': 'Free Litifilimab Plasma Concentration [µg/ml]', 'PD': 'Total BDCA2 Expression on pDCs [% Change]', 'PD_skin': 'Free BDCA2 Expression on pDCs [% Change]'}

activity_objects_dict = {}
# Create activity objects for each dose
for (model_key, model) in all_models.items():
    PK_data = all_datasets[model_key]['PK']
    bodyweight = bodyweights[model_key]
    activity_objects = {}
    for dose_key in PK_data.keys():
        activity = sund.Activity(time_unit='h')
        if 'IV_in' in PK_data[dose_key]['input']:
            activity.add_output(
                'piecewise_constant',
                "IV_in",
                t = PK_data[dose_key]['input']['IV_in']['t'],
                f = bodyweight * np.array(PK_data[dose_key]['input']['IV_in']['f'])
            )
        if 'SC_in' in PK_data[dose_key]['input']:
            activity.add_output(
                'piecewise_constant',
                "SC_in",
                t = PK_data[dose_key]['input']['SC_in']['t'],
                f = PK_data[dose_key]['input']['SC_in']['f']
            )
        activity_objects[dose_key] = activity
    activity_objects_dict[model_key] = activity_objects

# Create simulation objects for each dose
simulation_objects_dict = {}
for (model_key, model) in all_models.items():
    activity_objects = activity_objects_dict[model_key]
    simulation_objects = {}
    for dose_key, activity in activity_objects.items():
        simulation = sund.Simulation(models = model, activities = activity, time_unit = 'h')
        simulation_objects[dose_key] = simulation
    simulation_objects_dict[model_key] = simulation_objects

# Define a function to plot a dataset
def plot_dataset(data, measurement, ylabel, face_color='k'):
    plt.errorbar(data['time'], data[measurement], data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel(ylabel)

# Define a function to plot a simulation
def plot_sim(params, sims, timepoints, feature_to_plot, color='b'):
    sims.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sims.feature_names.index(feature_to_plot)
    plt.plot(sims.time_vector, sims.feature_data[:, feature_idx], color)

# Define the joint cost function for PK and PD data
def fcost_joint(params, sims, dataset):
    global measurements
    costs = {}
    for data_key, data in dataset.items():
        measurement = measurements[data_key]
        cost = 0
        for dose in data:
            try:
                sims[dose].simulate(time_vector=data[dose]['time'], parameter_values=params, reset=True)
                sim = sims[dose].feature_data[:, sims[dose].feature_names.index(f'{data_key}_sim')]
                y = data[dose][f"{measurement}"]
                SEM = data[dose]["SEM"]
                cost += np.sum(np.square(((sim - y) / SEM)))
            except Exception as e:
                if "CVODE" not in str(e):
                    print(f"Simulation of {dose} failed")
                    cost = 1e30
                    break
        costs[data_key] = cost
    return costs


# Initial parameter values for optimization
merged_initial_params = [0.713, 0.0096, 2.6, 1.125, 6.987, 4.368, 2.6, 0.0065, 0.0338, 0.081, 0.95, 0.8, 0.95, 0.45, 0.2, 0.00552, 0.00552, 0.28, 5.54, 2624]
initial_params_HV = np.delete(merged_initial_params.copy(), [11,16])  # Removing parameters for SLE clearance and RCS
initial_params_SLE = np.delete(merged_initial_params.copy(), [10,15])  # Removing parameter for HV clearance and RCS
all_initial_params = {'HV': initial_params_HV, 'SLE': initial_params_SLE}

for model_key in all_models.keys():
    print(f"Initial parameters for {model_key} model:", all_initial_params[model_key])

# Convert the initial parameters to logarithmic scale for optimization
merged_initial_params_log = np.log(merged_initial_params)

# Bounds for the parameters
# The bound factors are chosen to allow some flexibility in the optimization while keeping parameters physiologically reasonable
# Bounds for parameters which have reliable literature values are set to 1 (frozen parameters)
bound_factors = [1.25, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 10, 1, 1]


# Calculate the logarithmic bounds for the parameters
# The bounds are defined as log(initial_params) ± log(bound_factors)
lower_bounds = np.log(merged_initial_params) - np.log(bound_factors)
upper_bounds = np.log(merged_initial_params) + np.log(bound_factors)
bounds_log = Bounds(lower_bounds, upper_bounds)

# Print the bounds for the parameters
print("Lower bounds:", np.exp(lower_bounds))
print("Upper bounds:", np.exp(upper_bounds))

# Print cost for initial parameters

for model_key in all_models.keys():
    sims = simulation_objects_dict[model_key]
    params = all_initial_params[model_key]
    dataset = all_datasets[model_key]
    model_costs = fcost_joint(params, sims, dataset)
    for cost_key, cost_value in model_costs.items():
        print(f"Initial cost for {model_key} {cost_key}: {cost_value:.2f}")
    print(f"Total initial cost for {model_key}: {sum(model_costs.values()):.2f}")
    

# Calculate the degrees of freedom and chi2-limits for PK and PD separately
dgf = { 'HV': {}, 'SLE': {} }
chi2_limits = { 'HV': {}, 'SLE': {} }
for model_key in all_models.keys():
    dataset = all_datasets[model_key]
    for data_key, data in dataset.items():
        dgf[model_key][data_key] = sum(np.count_nonzero(np.isfinite(data[dose]["SEM"])) for dose in data)
        chi2_limits[model_key][data_key] = chi2.ppf(0.95, dgf[model_key][data_key])
        print(f"Chi2 limit for {model_key} {data_key}: {chi2_limits[model_key][data_key]:.3f}")

# Plot initial simulations against data for each model and simulation type
for model_key in all_models.keys():
    dataset = all_datasets[model_key]
    time_vectors = all_time_vectors[model_key]
    for (data_key, data) in dataset.items():
        time_vector = time_vectors[data_key]
        measurement = measurements[data_key]
        ylabel = ylabels[data_key]
        sims = simulation_objects_dict[model_key]
        params = all_initial_params[model_key]
        
        for dose in data:
            plt.figure()
            plot_sim(params, sims[dose], time_vector[dose], feature_to_plot=f"{data_key}_sim")
            plot_dataset(data[dose], measurement, ylabel)
            plt.title(f"{data_key} Simulation for {dose}")

# Show the plots
plt.show()

# Define a callback function to save the optimization results
# This function saves the current parameters and their cost to a JSON file
def callback(x, file_name):
    with open(f"./{file_name}.json",'w') as file:
        out = {"x": x}
        json.dump(out,file, cls=NumpyArrayEncoder)

# Define a callback function for logging the optimization progress
# This function converts the parameter to original scale and calls the callback function to save the parameters
def callback_log(x, file_name='temp'):
    callback(np.exp(x), file_name=file_name)

# Define a callback function for logging the evolution of the optimization
# This function is called at each iteration of the optimization and calls the callback_log function
def callback_evolution_log(x,convergence):
    callback_log(x, file_name='temp-evolution')

# Create the output directory for saving results
output_dir = '../../Results/Acceptable params'
os.makedirs(output_dir, exist_ok=True)
best_result_path = os.path.join(output_dir, 'best_result_80_pdc_mm2.json')

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
acceptable_params_path = os.path.join(output_dir, 'acceptable_params_80_pdc_mm2.json')
if os.path.exists(acceptable_params_path) and os.path.getsize(acceptable_params_path) > 0:
    with open(acceptable_params_path, 'r') as f:
        acceptable_params = json.load(f)

# Define the cost function arguments
cost_function_args = (simulation_objects_dict, all_datasets)

def fcost_uncertainty(merged_params_log, simulation_objects_dict, all_datasets):
    global acceptable_params
    global best_cost
    global best_param

    merged_params = np.exp(merged_params_log)
    HV_params = np.delete(merged_params.copy(), [11,16])  # Removing parameters for SLE clearance and RCS
    SLE_params = np.delete(merged_params.copy(), [10,15])  # Removing parameter for HV clearance and RCS
    all_params = {'HV': HV_params, 'SLE': SLE_params}

    params_pass = True
    all_costs = {}
    total_cost = 0.0

    for model_key in all_models.keys():
        params = all_params[model_key]
        sims = simulation_objects_dict[model_key]
        dataset = all_datasets[model_key]
        model_costs = fcost_joint(params, sims, dataset)
        for data_key in dataset.keys():
            if model_costs[data_key] < chi2_limits[model_key][data_key]:
                continue
            else:
                params_pass = False
        all_costs[model_key] = model_costs
        total_cost += sum(model_costs.values())


    if params_pass == True:
        acceptable_params.append(merged_params.tolist())
        if total_cost < best_cost:
            best_cost = total_cost
            best_param = merged_params.copy()
            print(f"New best total cost: {best_cost}")
            for model_key in all_models.keys():
                model_costs = all_costs[model_key]
                for data_key, cost_value in model_costs.items():
                    print(f"  {model_key} {data_key} cost: {cost_value:.2f}")

    return total_cost

# Perform optimization using differential evolution
# This optimization runs multiple iterations to find the best parameters that minimize the cost function
for i in range(5):  # Run the optimization 5 times
    res = differential_evolution(
        func=fcost_uncertainty,
        bounds=bounds_log,
        args=cost_function_args,
        x0=merged_initial_params_log,
        callback=callback_evolution_log,
        disp=True
    )

# Save all acceptable parameter sets to a CSV file
with open(os.path.join(output_dir, 'acceptable_params_80_pdc_mm2.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(acceptable_params)

# Save all acceptable parameter sets to a JSON file
with open(os.path.join(output_dir, 'acceptable_params_80_pdc_mm2.json'), 'w') as f:
    json.dump(acceptable_params, f, cls=NumpyArrayEncoder)

# Save the best parameter set to a JSON file
with open(os.path.join(output_dir, 'best_result_80_pdc_mm2.json'), 'w') as f:
    json.dump({'best_cost': best_cost, 'best_param': best_param.tolist()}, f, cls=NumpyArrayEncoder)

# print the number of acceptable parameter sets collected
print(f"Number of acceptable parameter sets collected: {len(acceptable_params)}")

# Define a function to plot uncertainty of PK and PD in the model
# This function randomly selects a subset of 500 acceptable parameter sets and plots their simulations against PK and PD data
def plot_uncertainty(acceptable_params, simulation_objects_dict, all_datasets, n_params_to_plot=500):
    random.shuffle(acceptable_params)

    for model_key in all_models.keys():
        dataset = all_datasets[model_key]
        time_vectors = all_time_vectors[model_key]
        for (data_key, data) in dataset.items():
            time_vector = time_vectors[data_key]
            measurement = measurements[data_key]
            ylabel = ylabels[data_key]
            sims = simulation_objects_dict[model_key]
            
            for dose in data:
                plt.figure()

                # Keep track of successful and failed simulations
                success_count = 0
                fail_count = 0

                # Plot parameter sets until n_params_to_plot is reached or all parameters are exhausted
                for param in acceptable_params:
                    if success_count >= n_params_to_plot:
                        break
                    try:
                        if model_key == 'HV':
                            HV_params = np.delete(param.copy(), [11,16])  # Removing parameters for SLE clearance, kdegs and kints
                            plot_sim(HV_params, sims[dose], time_vector[dose], feature_to_plot=f"{data_key}_sim")
                        else:
                            SLE_params = np.delete(param.copy(), [10,15])  # Removing parameter for HV clearance
                            plot_sim(SLE_params, sims[dose], time_vector[dose], feature_to_plot=f"{data_key}_sim")

                        success_count += 1
                    except RuntimeError as e:
                        if "CVODE" in str(e):
                            fail_count += 1
                            continue
                        else:
                            raise e

                # Plot dataset and title   
                plot_dataset(data[dose], measurement, ylabel)
                plt.title(f"{data_key} Simulations for {dose}")

                # Print the number of successful and failed simulations
                print(f"  Successful simulations for {dose}: {success_count}")
                print(f"  Failed (CVODE) simulations for {dose}: {fail_count}")

# Plot the uncertainty of the model simulations using the acceptable parameters
plot_uncertainty(acceptable_params, simulation_objects_dict, all_datasets)

# Show the plots
plt.show()
