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
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("../../Data/SLE_PD_data.json", "r") as f:
    PD_data = json.load(f)

# Define a function to plot one PD_dataset
def plot_PD_dataset(PD_data, face_color='k'):
    plt.errorbar(PD_data['time'], PD_data['BDCA2_median'], PD_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BDCA2 levels on pDCs, percentage change from baseline')

# Definition of the function that plots all PD_datasets
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
bodyweight = 70

# Create activity objects for each dose
IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_20_SLE']['input']['IV_in']['f']))

# Create simulation objects for each dose
model_sims = {
    'IVdose_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
}

# Create time vectors for each experiment
# The time vectors are created based on the maximum time in the PD_data for each experiment
time_vectors = {exp: np.arange(-10, PD_data[exp]["time"][-1] + 0.01, 1) for exp in PD_data}

# Define a function to plot the simulation results
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PD_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Define a function to plot the simulation results with PD data
def plot_sim_with_PD_data(params, sims, PD_data, color='b'):
    for experiment in PD_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_PD_dataset(PD_data[experiment])
        plot_sim(params, sims[experiment], timepoints, color)
        plt.title(experiment)

# Define the cost function for the optimization
# This function calculates the cost based on the difference between the simulated and observed PD data
def fcost(params, sims, PD_data):
    cost = 0
    for dose in PD_data:
        try:
            sims[dose].simulate(time_vector = PD_data[dose]["time"], parameter_values = params, reset = True)
            PD_sim = sims[dose].feature_data[:,0]
            y = PD_data[dose]["BDCA2_median"]
            SEM = PD_data[dose]["SEM"] 
            cost += np.sum(np.square(((PD_sim - y) / SEM)))
        except Exception as e:
            if "CVODE" not in str(e):
                print(str(e))
            return 1e30
    return cost

# Define the initial guesses for the parameters
initial_params = [0.818, 0.011, 2.6, 1.81, 6.3, 4.37, 2.6, 0.0103, 0.0296, 0.081, 0.77, 0.95, 0.605, 0.2, 5.5e-3, 120, 3612, 0.91, 4.4e-4, 0.04, 1, 14000]
cost_PD = fcost(initial_params, model_sims, PD_data)
print(f"Cost of the PD model: {cost_PD}")

# Calculate the degrees of freedom for the chi-squared test
dgf = sum(np.count_nonzero(np.isfinite(PD_data[exp]["SEM"])) for exp in PD_data)
chi2_limit = chi2.ppf(0.95, dgf)

# Print the chi-squared limit and whether the cost exceeds it
print(f"Chi2 limit: {chi2_limit}")
print(f"Cost > limit (rejected?): {cost_PD > chi2_limit}")

# Plot the simulation results with PD data
plot_sim_with_PD_data(initial_params, model_sims, PD_data)

# Show the plots
plt.show()

# Define a callback function to save the optimization results
# This function saves the current parameters and their cost to a JSON file
def callback(x, file_name):
    with open(f"./{file_name}.json",'w') as file:
        out = {"x": x}
        json.dump(out,file, cls=NumpyArrayEncoder)

# Define the cost function arguments
# This is a tuple containing the model simulations and PD data
cost_function_args = (model_sims, PD_data)

# Convert the initial parameters to logarithmic scale for optimization
initial_params_log = np.log(initial_params)

# Bounds for the parameters
# The bound factors are chosen to allow some flexibility in the optimization while keeping parameters physiologically reasonable
# Bounds for parameters which have reliable literature values are set to 1 (frozen parameters)
# Additionally, since this is an optimization of PD, parameters related to PD are also frozen
bound_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]  

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
def fcost_log(params_log, sims, PD_data):
    return fcost(np.exp(params_log.copy()), sims, PD_data)     

# Define a callback function for logging the optimization progress
# This function converts the parameter to original scale and calls the callback function to save the parameters
def callback_log_PD(x, file_name='SLE-PD-temp'):
    callback(np.exp(x), file_name=file_name)

# Define a callback function for logging the evolution of the optimization
# This function is called at each iteration of the optimization and calls the callback_log_PD function
def callback_PD_evolution_log(x,convergence):
    callback_log_PD(x, file_name='SLE-PD-temp-evolution')

output_dir = '../../Results/Acceptable params'
os.makedirs(output_dir, exist_ok=True)
best_result_path = os.path.join(output_dir, 'best_SLE_PD_result.json')

# Load previous best result if available
if os.path.exists(best_result_path) and os.path.getsize(best_result_path) > 0:
    with open(best_result_path, 'r') as f:
        best_data = json.load(f)
        best_cost_PD = best_data['best_cost']
        best_param_PD = np.array(best_data['best_param'])
else:
    best_cost_PD = np.inf
    best_param_PD = None

# Create list to store all acceptable parameter sets
acceptable_params_PD = []

# Define the cost function for uncertainty analysis
# This function calculates the cost and updates the acceptable parameters and best cost if the cost is improved
def fcost_uncertainty_PD(param_log, model, PD_data):
    global acceptable_params_PD
    global best_cost_PD
    global best_param_PD

    # Convert the logarithmic parameters to original scale and calculate the cost using the fcost function
    params = np.exp(param_log)
    cost = fcost(params, model, PD_data)

    # Save all parameter sets with cost < chi2_limit
    if cost < chi2_limit:
        acceptable_params_PD.append(params)

    # Update the best cost and parameter set
    if cost < best_cost_PD:
        best_cost_PD = cost
        best_param_PD = params.copy()
        print(f"New best cost: {best_cost_PD}")

    return cost

# Perform optimization using differential evolution
# This optimization runs multiple iterations to find the best parameters that minimize the cost function
for i in range(5):
    res = differential_evolution(
        func=fcost_uncertainty_PD,
        bounds=bounds_log,
        args=cost_function_args,
        x0=initial_params_log,
        callback=callback_PD_evolution_log,
        disp=True
    )
    initial_params_log = res['x']

# Save all acceptable parameter sets to a CSV file
with open(os.path.join(output_dir, 'acceptable_params_SLE_PD.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(acceptable_params_PD)

# Save all acceptable parameter sets to a JSON file
with open(os.path.join(output_dir, 'acceptable_params_SLE_PD.json'), 'w') as f:
    json.dump(acceptable_params_PD, f, cls=NumpyArrayEncoder)

# Save the best parameter set to a JSON file
with open(os.path.join(output_dir, 'best_SLE_PD_result.json'), 'w') as f:
    json.dump({'best_cost': best_cost_PD, 'best_param': best_param_PD.tolist()}, f, cls=NumpyArrayEncoder)

# print the number of acceptable parameter sets collected
print(f"Number of acceptable parameter sets collected: {len(acceptable_params_PD)}")

# Define a function to plot uncertainty of PD in the model
# This function randomly selects a subset of 500 acceptable parameter sets and plots their simulations against the PD data
def plot_uncertainty(all_params, sims, PD_data, color='b', n_params_to_plot=500):
    random.shuffle(all_params)

    for experiment in PD_data:
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

        # Plot the PD data and title for the experiment
        plot_PD_dataset(PD_data[experiment])
        plt.title(experiment)

        # Print the number of successful and failed simulations
        print(f"  Successful simulations: {success_count}")
        print(f"  Failed (CVODE) simulations: {fail_count}")

# Plot the uncertainty of PD in the model using the acceptable parameters
plot_uncertainty(acceptable_params_PD, model_sims, PD_data)

# Show the plots
plt.show()
