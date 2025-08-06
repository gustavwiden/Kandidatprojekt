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
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Open the PK data file and read its contents
with open("../../Data/SLE_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Open the PD data file and read its contents
with open("../../Data/SLE_PD_data.json", "r") as f:
    PD_data = json.load(f)

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
# The time vectors are created based on the maximum time in the PK_data and PD_data for each experiment
time_vectors_PK = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}
time_vectors_PD = {exp: np.arange(-10, PD_data[exp]["time"][-1] + 0.01, 1) for exp in PD_data}

# Define the joint cost function for the optimization
# This function calculates the cost based on the difference between simulations and PK/PD data
def fcost_joint(params, sims, PK_data, PD_data, pk_weight=1.0, pd_weight=0.0):
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
initial_params = [0.5982467918487137, 0.013501146489749132, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.75, 0.95, 0.7467544604963505, 0.2, 0.00549200604682213, 0.9621937056820449, 0.1, 5.539999999999999, 5.539999999999999, 2623.9999999999995]


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
bound_factors = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1]

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
best_result_path = os.path.join(output_dir, 'best_SLE_result.json')

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
acceptable_params_path = os.path.join(output_dir, 'acceptable_params_SLE.json')
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
    if pk_cost < chi2_limit_PK:
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

# print the number of acceptable parameter sets collected
print(f"Number of acceptable parameter sets collected: {len(acceptable_params)}")
# define objective function for pypesto 

sampling_params_SLE = []

best_param = np.array(best_param)

def proxy_f(params):
    return fcost_sampling(params, model_sims, PK_data, PD_data, True)

def insert_params(selected_params, best_param, selected_indices):
    full_param = best_param.copy()
    full_param[selected_indices] = selected_params
    return full_param


def fcost_sampling(params_reduced, model_sims, PK_data, PD_data, SaveParams):
    global best_param
    global best_cost

    selected_indices = [14]
    full_param = insert_params(params_reduced, best_param, selected_indices)
    joint_cost, pk_cost, pd_cost = fcost_joint(full_param, model_sims, PK_data, PD_data)

    if SaveParams and pk_cost < chi2_limit_PK:
        sampling_params_SLE.append(full_param)

        if joint_cost < best_cost:
            best_cost = joint_cost
            best_param = np.array(full_param.copy())
            print(f"New best joint cost: {best_cost} (PK: {pk_cost}, PD: {pd_cost})")

    return joint_cost


# Optimization using pypesto ------------------------------------------------------------
lb=np.array(np.exp(lower_bounds))[[14]]
ub=np.array(np.exp(upper_bounds))[[14]]

params_for_MCMC = best_param[[14]]

parameter_scales = ['lin']*len(params_for_MCMC)
parameter_names = ['CL']

# MCMC sampling using pypesto ------------------------------------------------------------
# Define a custom objective function for sampling
import pypesto.sample as sample
custom_objective = pypesto.Objective(fun=proxy_f, grad = None, hess = None, hessp = None)
custom_problem = pypesto.Problem(objective=custom_objective, lb=lb, ub=ub, x_guesses=[params_for_MCMC], x_scales=parameter_scales, x_names = parameter_names)

n_samples = int(1e6)
print(params_for_MCMC)
sampler = sample.AdaptiveMetropolisSampler()

result_sampling = sample.sample(
    problem=custom_problem, n_samples=n_samples, sampler=sampler, result=None, x0=params_for_MCMC)

with open("sampling_result_SLE.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    samples = np.array(result_sampling.sample_result["trace_x"])[0]
    writer.writerows(samples)



trace_array = np.loadtxt("sampling_result_SLE.csv", delimiter=",")  # loads your samples

if trace_array.ndim > 1:
    trace_array = trace_array.flatten()

plt.figure(figsize=(6, 4))
plt.hist(trace_array, bins='auto', color='green')
plt.title(parameter_names[0])
plt.ylabel('Frequency')
plt.xlabel('Parameter value')

formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().tick_params(axis='x', labelrotation=45, labelsize=8)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

plt.tight_layout()
plt.show()



# from scipy.optimize import minimize, differential_evolution
# import numpy as np

# def fcost_PL(param_log, model_sims, PK_data, PD_data, param_index, PL_revValue):
#     params = np.exp(param_log)
#     joint_cost, pk_cost, pd_cost = fcost_joint(params, model_sims, PK_data, PD_data)

#     # Profile penalty in log-space
#     penalty = 1e6 * (param_log[param_index] - PL_revValue)**2
#     return joint_cost + penalty

# # ---- Profile Likelihood Loop ----
# parameterIdx = 14  
# nSteps = 40
# step_size = 0.005
# best_param_log = np.log(best_param)

# PL_revValues = np.zeros(nSteps * 2)
# PL_costs = np.zeros(nSteps * 2)

# count = 0
# max_retries = 3  # Number of times to retry with perturbed initial guess if optimization fails

# for direction in [-1, 1]:
#     for step in range(nSteps):
#         PL_revValue = best_param_log[parameterIdx] + direction * step_size * step
#         PL_revValues[count] = PL_revValue

#         success = False
#         retry = 0

#         while not success and retry <= max_retries:
#             # Start from perturbed log-params
#             x0 = best_param_log + np.random.normal(0, 0.01, size=best_param_log.shape)
#             x0[parameterIdx] = PL_revValue  # Keep profiled parameter fixed

#             res = minimize(
#                 fun=fcost_PL,
#                 x0=x0,
#                 args=(model_sims, PK_data, PD_data, parameterIdx, PL_revValue),
#                 method='L-BFGS-B',
#                 bounds=bounds_log,
#                 options={'disp': False, 'maxiter': 1000}
#             )

#             success = res.success
#             retry += 1

#         # Fallback: use differential evolution if local fails repeatedly
#         if not success:
#             print(f"Local optimization failed after {max_retries} retries at step {count}. Trying global optimization.")
#             def wrapper_de(p):
#                 return fcost_PL(p, model_sims, PK_data, PD_data, parameterIdx, PL_revValue)

#             result = differential_evolution(
#                 func=wrapper_de,
#                 bounds=bounds_log,
#                 strategy='best1bin',
#                 maxiter=20,
#                 popsize=10,
#                 polish=True,
#                 disp=False
#             )
#             PL_costs[count] = result.fun
#         else:
#             PL_costs[count] = res.fun

#         count += 1

# # Reorder profile for plotting
# PL_revValues[0:nSteps] = PL_revValues[-(nSteps+1):-((2*nSteps)+1):-1]
# PL_costs[0:nSteps] = PL_costs[-(nSteps+1):-((2*nSteps)+1):-1]

# # Plot
# plt.figure()
# plt.plot(np.exp(PL_revValues), PL_costs, linestyle='--', marker='o', label='PL (fast)', color='k')
# plt.axhline(y=chi2.ppf(0.95, dgf_PK + dgf_PD), linestyle='--', color='r', label='Chi² threshold')
# plt.xlabel('Parameter value')
# plt.ylabel('Objective function value')
# plt.ylim(0, 100)
# plt.legend()
# plt.tight_layout()
# plt.show()
