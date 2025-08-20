# Import necessary libraries
import os
import re
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sund
import json
from scipy.stats import chi2
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import odeint
import sys
import csv
import random
import requests
import pypesto
import pypesto.optimize as optimize
import pypesto.sample as sample

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Load PK data
with open("../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load PD data
with open("../../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

# Install and load the model
sund.install_model('../../Models/mPBPK_model.txt')
print(sund.installed_models())
model = sund.load_model("mPBPK_model")

# Average bodyweight for healthy volunteers (HV) (cohort 1-7 in the phase 1 trial)
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
initial_params = [0.713, 0.0096, 2.6, 1.125, 6.987, 4.368, 2.6, 0.0065, 0.0338, 0.081, 0.95, 0.95, 0.45, 0.2, 0.00552, 1, 5.54, 2624]

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
cost_function_args = (model_sims, PK_data, PD_data)

# Convert the initial parameters to logarithmic scale for optimization
initial_params_log = np.log(initial_params)

# Bounds for the parameters were doubled to allow for more flexibility than in optimization
bound_factors = [2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 10, 1, 1]

# Calculate the logarithmic bounds for the parameters
# The bounds are defined as log(initial_params) ± log(bound_factors)
lower_bounds = np.log(initial_params) - np.log(bound_factors)
upper_bounds = np.log(initial_params) + np.log(bound_factors)
bounds_log = Bounds(lower_bounds, upper_bounds)

# Print the bounds for the parameters
print("Lower bounds:", np.exp(lower_bounds))
print("Upper bounds:", np.exp(upper_bounds))

# Define a callback function for logging the optimization progress
# This function converts the parameter to original scale and calls the callback function to save the parameters
def callback_log(x, file_name='temp'):
    callback(np.exp(x), file_name=file_name)

# Define a callback function for logging the evolution of the optimization
# This function is called at each iteration of the optimization and calls the callback_log_PK function
def callback_evolution_log(x,convergence):
    callback_log(x, file_name='temp-evolution')

# Create output directory for best results
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

# Define the cost function for optimization
# This function calculates the cost based on the joint cost, PK cost, and PD cost
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

# THE OPTIMIZATION IS NOT RUN IN THIS SCRIPT, BUT IN OPTIMIZE_HV.py
# HOWEVER, THE CODE IS LEFT HERE IN CASE IT IS NEEDED FOR FUTURE REFERENCE

# for i in range(5):  # Run the optimization 5 times
#     res = differential_evolution(
#         func=fcost_optimization,
#         bounds=bounds_log,
#         args = cost_function_args,
#         x0=initial_params_log,
#         callback=callback_evolution_log,
#         disp=True
#     )

# Save all acceptable parameter sets to a CSV file
with open(os.path.join(output_dir, 'acceptable_params.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(acceptable_params)

# Save all acceptable parameter sets to a JSON file
with open(os.path.join(output_dir, 'acceptable_params.json'), 'w') as f:
    json.dump(acceptable_params, f, cls=NumpyArrayEncoder)

# Save the best parameter set to a JSON file
with open(os.path.join(output_dir, 'best_result.json'), 'w') as f:
    json.dump({'best_cost': best_cost, 'best_param': best_param.tolist()}, f, cls=NumpyArrayEncoder)

# print the number of acceptable parameter sets collected
print(f"Number of acceptable parameter sets collected: {len(acceptable_params)}")


# MCMC sampling begins here ------------------------------------------------------------- 
sampling_params = []
best_param = np.array(best_param)

# Define a proxy function for the cost function to be used in MCMC sampling
def proxy_f(params):
    return fcost_sampling(params, model_sims, PK_data, PD_data, True)

# Define a function to restore the full parameter set from the reduced parameters used in MCMC sampling
def insert_params(selected_params, best_param, selected_indices):
    full_param = best_param.copy()
    full_param[selected_indices] = selected_params
    return full_param

# Define the cost function for MCMC sampling
# This function calculates the cost based on the joint cost, PK cost, and PD cost
def fcost_sampling(params_reduced, model_sims, PK_data, PD_data, SaveParams):
    global best_param
    global best_cost

    selected_indices = [0, 1, 12, 14, 15]
    full_param = insert_params(params_reduced, best_param, selected_indices)
    joint_cost, pk_cost, pd_cost = fcost_joint(full_param, model_sims, PK_data, PD_data)

    # Only save parameters if they are below BOTH chi2 limits
    if SaveParams and pk_cost < chi2_limit_PK and pd_cost < chi2_limit_PD:
        sampling_params.append(full_param)

        if joint_cost < best_cost:
            best_cost = joint_cost
            best_param = np.array(full_param.copy())
            print(f"New best joint cost: {best_cost} (PK: {pk_cost}, PD: {pd_cost})")

    return joint_cost

# Define the lower and upper bounds for the parameters used in MCMC sampling
lb=np.array(np.exp(lower_bounds))[[0, 1, 12, 14, 15]]
ub=np.array(np.exp(upper_bounds))[[0, 1, 12, 14, 15]]

# Define the parameters to be used in MCMC sampling
params_for_MCMC = best_param[[0, 1, 12, 14, 15]]
print(params_for_MCMC)

# Define the parameter scales and names for MCMC sampling
parameter_scales = ['lin']*len(params_for_MCMC)
parameter_names = ['F', 'ka',  'RC2',  'CL', 'kdeg']

# Create a custom objective and create a custom problem for MCMC sampling
custom_objective = pypesto.Objective(fun=proxy_f, grad = None, hess = None, hessp = None)
custom_problem = pypesto.Problem(objective=custom_objective, lb=lb, ub=ub, x_guesses=[params_for_MCMC], x_scales=parameter_scales, x_names = parameter_names)

# Set the number of samples and the sampler for MCMC sampling
n_samples = int(1e6)
sampler = sample.AdaptiveMetropolisSampler()

# Run the MCMC sampling
result_sampling = sample.sample(
    problem=custom_problem, n_samples=n_samples, sampler=sampler, result=None, x0=params_for_MCMC)

# Save the sampling results to a CSV file
with open(os.path.join(output_dir, "sampling_result.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    samples = np.array(result_sampling.sample_result["trace_x"])[0]
    writer.writerows(samples)

# MCMC sampling ends here -------------------------------------------------------------


# Plot the results!

# Load the sampling results from the CSV file
trace_array = np.loadtxt(os.path.join(output_dir, "sampling_result.csv"), delimiter=",")

# Create a figure with subplots for each parameter
rows, cols = 2, 3
fig, axs = plt.subplots(rows, cols, figsize=(8, 6))
axes = axs.flatten()

# Adjust the number of subplots based on the number of parameters
num_params = trace_array.shape[1]

# Plot histograms for each parameter
for i in range(num_params):
    ax = axes[i]
    data_for_hist = trace_array[:, i]
    ax.hist(data_for_hist, bins='auto', color='green')
    ax.set_title(parameter_names[i])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Parameter value')

    # Set x-axis to scientific notation
    formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', labelrotation=45, labelsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

# Hide any unused subplots
for j in range(num_params, len(axes)):
    fig.delaxes(axes[j])

# Save the histogram
save_dir = "../../Results/Validation"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "MCMC mPBPK-model.png")
plt.savefig(save_path, format='png', dpi=600)

plt.tight_layout()
plt.show()


# Profile Likelihood (PL) begins here ---------------------------------------------------

# Define the cost function for PL which calls fcost_joint and punishes values too far from the best value
def fcost_PL(param_log, model_sims, PK_data, PD_data, param_index, PL_revValue):
    params = np.exp(param_log)
    joint_cost, pk_cost, pd_cost = fcost_joint(params, model_sims, PK_data, PD_data)
    penalty = 1e6 * (param_log[param_index] - PL_revValue)**2
    return joint_cost + penalty

# Create list to store all acceptable parametersets foudn during PL
PL_params = []

# These settings are changed depending on which parameter is being analysed
parameterIdx = 1  
nSteps = 50
step_size = 0.04
best_param_log = np.log(best_param)

# To store values and costs
PL_revValues = np.zeros(nSteps * 2)
PL_costs = np.zeros(nSteps * 2)

# The loop below uses the minimize optimizer to reduce run time, but if a problem occurs, it has differential solver to fall back on
count = 0

 # This loop takes n steps in each direction with the given step size
for direction in [-1, 1]:
    x_opt_prev = best_param_log.copy()
    for step in range(nSteps):
        PL_revValue = best_param_log[parameterIdx] + direction * step_size * step
        PL_revValues[count] = PL_revValue

        # Use the latest value as a warm-start
        x0 = x_opt_prev.copy()
        x0[parameterIdx] = PL_revValue

        success = False

        # Use the minimize-optimizer
        res = minimize(
            fun=fcost_PL,
            x0=x0,
            args=(model_sims, PK_data, PD_data, parameterIdx, PL_revValue),
            method='L-BFGS-B',
            bounds=bounds_log,
            options={'disp': False, 'maxiter': 1000}
        )

        success = res.success

        # Fallback: use differential evolution if minimize fails
        if not success:
            print(f"Optimization failed on step {count}. Trying differential evolution.")

            # Use differential evolution
            result = differential_evolution(
                func=fcost_PL,
                bounds=bounds_log,
                args=(model_sims, PK_data, PD_data, parameterIdx, PL_revValue)
                strategy='best1bin',
                maxiter=20,
                popsize=10,
                polish=True,
                disp=False
            )

            # Save cost of the step and update x_opt_prev to use warm start
            PL_costs[count] = result.fun
            x_opt_prev = result.x.copy()
            step_params = np.exp(result.x)
        else:
            # Same but if minimize works
            PL_costs[count] = res.fun
            x_opt_prev = res.x.copy()
            step_params = np.exp(res.x)

        # Caluculate cost for the params found during the step and only save them to PL_params if they are below both chi2-limits
        joint_cost, pk_cost, pd_cost = fcost_joint(step_params, model_sims, PK_data, PD_data)
        if pk_cost < chi2_limit_PK and pd_cost < chi2_limit_PD:
            PL_params.append(step_params.tolist())
        count += 1

# Save all acceptable params form PL to a csv file
with open(os.path.join(output_dir, "acceptable_params_PL_ka.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(PL_params)


# Reorder values for plotting
PL_revValues[0:nSteps] = PL_revValues[-(nSteps+1):-((2*nSteps)+1):-1]
PL_costs[0:nSteps] = PL_costs[-(nSteps+1):-((2*nSteps)+1):-1]

# Plot all steps and its cost
plt.figure()
plt.plot(np.exp(PL_revValues), PL_costs, linestyle='--', marker='o', label='PL (fast)', color='k')
plt.axhline(y=chi2.ppf(0.95, dgf_PK + dgf_PD), linestyle='--', color='r', label='Chi² threshold')
plt.xlabel('Parameter value')
plt.ylabel('Objective function value')
plt.ylim(0, 300)
plt.legend()

# Save plot
save_dir = "../../Results/Validation"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "PL_ka.png")
plt.savefig(save_path, format='png', dpi=600)

plt.tight_layout()
plt.show()
