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
with open("../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../Data/PK_data.json", "r") as f:
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

# Install and load the model
sund.install_model('../../Models/mPBPK_model.txt')
print(sund.installed_models())
model = sund.load_model("mPBPK_model")

# Bodyweight for subject in kg
bodyweight = 70

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
# The time vectors are created based on the maximum time in the PK_data for each experiment
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

# Define a function to plot the simulation results
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PK_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Define a function to plot the simulation results with PK data
def plot_sim_with_PK_data(params, sims, PK_data, color='b'):
    for experiment in PK_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_PK_dataset(PK_data[experiment])
        plot_sim(params, sims[experiment], timepoints, color)
        plt.title(experiment)

# Define the cost function for the optimization
# This function calculates the cost based on the difference between the simulated and observed PK data
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

# Define the initial guesses for the parameters
initial_params = [0.713, 0.00975, 2.6, 1.81, 6.3, 4.37, 2.6, 1.03e-2, 2.96e-2, 8.1e-2, 0.7, 0.95, 0.55, 0.2, 5.52e-3, 73, 3731, 2.05, 4.4e-4, 1, 1, 14000]

# Print cost for initial parameters
cost_PK = fcost(initial_params, model_sims, PK_data)
print(f"Cost of the PK model: {cost_PK}")

# Calculate the degrees of freedom for the chi-squared test
dgf = sum(np.count_nonzero(np.isfinite(PK_data[exp]["SEM"])) for exp in PK_data)
chi2_limit = chi2.ppf(0.95, dgf)

# Print the chi-squared limit and whether the cost exceeds it
print(f"Chi2 limit: {chi2_limit}")
print(f"Cost > limit (rejected?): {cost_PK > chi2_limit}")

# Plot the simulation results with PK data
plot_sim_with_PK_data(initial_params, model_sims, PK_data)

# Show the plots
plt.show()

# Define a callback function to save the optimization results
# This function saves the current parameters and their cost to a JSON file
def callback(x, file_name):
    with open(f"./{file_name}.json",'w') as file:
        out = {"x": x}
        json.dump(out,file, cls=NumpyArrayEncoder)

# Define the cost function arguments
# This is a tuple containing the model simulations and PK data
cost_function_args = (model_sims, PK_data)

# Convert the initial parameters to logarithmic scale for optimization
initial_params_log = np.log(initial_params)

# Bounds for the parameters
# The bound factors are chosen to allow some flexibility in the optimization while keeping parameters physiologically reasonable
# Bounds for parameters which have reliable literature values are set to 1 (frozen parameters)
# Additionally, since this is an optimization of PK, parameters related to PD are also frozen
bound_factors = [1.2, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1.1, 1, 1.1, 1, 2, 2, 4, 1, 1, 1, 1, 1]

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
def fcost_log(params_log, sims, PK_data):
    return fcost(np.exp(params_log.copy()), sims, PK_data)     

# Define a callback function for logging the optimization progress
# This function converts the parameter to original scale and calls the callback function to save the parameters
def callback_log_PK(x, file_name='PK-temp'):
    callback(np.exp(x), file_name=file_name)

# Define a callback function for logging the evolution of the optimization
# This function is called at each iteration of the optimization and calls the callback_log_PK function
def callback_PK_evolution_log(x,convergence):
    callback_log_PK(x, file_name='PK-temp-evolution')

output_dir = '../../Results/Acceptable params'
os.makedirs(output_dir, exist_ok=True)
best_result_path = os.path.join(output_dir, 'best_PK_result.json')

# Load previous best result if available
if os.path.exists(best_result_path) and os.path.getsize(best_result_path) > 0:
    with open(best_result_path, 'r') as f:
        best_data = json.load(f)
        best_cost_PK = best_data['best_cost']
        best_param_PK = np.array(best_data['best_param'])
else:
    best_cost_PK = np.inf
    best_param_PK = None

# Create list to store all acceptable parameter sets
acceptable_params_PK = []

# Define the cost function for uncertainty analysis
# This function calculates the cost and updates the acceptable parameters and best cost if the cost is improved
def fcost_uncertainty_PK(param_log, model, PK_data):
    global acceptable_params_PK
    global best_cost_PK
    global best_param_PK

    # Convert the logarithmic parameters to original scale and calculate the cost using the fcost function
    params = np.exp(param_log)
    cost = fcost(params, model, PK_data)

    # Save all parameter sets with cost < chi2_limit
    if cost < chi2_limit:
        acceptable_params_PK.append(params)

    # Update the best cost and parameter set
    if cost < best_cost_PK:
        best_cost_PK = cost
        best_param_PK = params.copy()
        print(f"New best cost: {best_cost_PK}")

    return cost

# Perform optimization using differential evolution
# This optimization runs multiple iterations to find the best parameters that minimize the cost function
for i in range(5):
    res = differential_evolution(
        func=fcost_uncertainty_PK,
        bounds=bounds_log,
        args=cost_function_args,
        x0=initial_params_log,
        callback=callback_PK_evolution_log,
        disp=True
    )
    initial_params_log = res['x']

# Save all acceptable parameter sets to a CSV file
with open(os.path.join(output_dir, 'acceptable_params_PK.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(acceptable_params_PK)

# Save all acceptable parameter sets to a JSON file
with open(os.path.join(output_dir, 'acceptable_params_PK.json'), 'w') as f:
    json.dump(acceptable_params_PK, f, cls=NumpyArrayEncoder)

# Save the best parameter set to a JSON file
with open(os.path.join(output_dir, 'best_PK_result.json'), 'w') as f:
    json.dump({'best_cost': best_cost_PK, 'best_param': best_param_PK.tolist()}, f, cls=NumpyArrayEncoder)

# print the number of acceptable parameter sets collected
print(f"Number of acceptable parameter sets collected: {len(acceptable_params_PK)}")

# Define a function to plot uncertainty of PK in the model
# This function randomly selects a subset of 500 acceptable parameter sets and plots their simulations against the PK data
def plot_uncertainty(all_params, sims, PK_data, color='b', n_params_to_plot=500):
    random.shuffle(all_params)

    for experiment in PK_data:
        print(f"\nPlotting uncertainty for: {experiment}")
        plt.figure()
       
        timepoints = time_vectors[experiment]

        # Keep track of successful and failed simulations
        success_count = 0
        fail_count = 0

        # Plot parameter sets until n_params_to_plot is reached or all parameters are exhausted
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

        # Plot the PK data and title for the experiment
        plot_PK_dataset(PK_data[experiment])
        plt.title(experiment)

        # Print the number of successful and failed simulations
        print(f"  Successful simulations: {success_count}")
        print(f"  Failed (CVODE) simulations: {fail_count}")

# Plot the uncertainty of PK in the model using the acceptable parameters
plot_uncertainty(acceptable_params_PK, model_sims, PK_data)

# Show the plots
plt.show()
