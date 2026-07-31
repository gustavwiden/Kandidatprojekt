import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import Bounds, differential_evolution
import csv
import random
import json

# Global setting of skin pDC density (pDCs/mm²)
# To run the parameter estimation for the other pDC densities, simply change between '1', '80' and '400'
pDC_density = '80'

from utils import base_dir, NumpyArrayEncoder, load_data, load_models, create_simulation_objects, plot_dataset, plot_sim, fcost_joint, merged_to_model_params, evaluate_cost, callback_evolution_log

data = load_data('HV_PK_data', 'HV_PD_data', 'SLE_PK_data', 'SLE_PD_data')

models = load_models('HV_model', f'SLE_model_{pDC_density}')

all_datasets = {'HV': {'PK': data['HV_PK_data'], 'PD': data['HV_PD_data']},
                'SLE': {'PK': data['SLE_PK_data'], 'PD': data['SLE_PD_data']}}

# Average bodyweight (kg) for HV and SLE patients (cohort 1-7 and cohort 8 respectively in the phase 1 trial)
bodyweights = {'HV': 73, 'SLE': 69}

time_vectors_PK = {dose: np.arange(-10, data['HV_PK_data'][dose]["time"][-1] + 0.01, 1) for dose in data['HV_PK_data']}
time_vectors_PD = {dose: np.arange(-10, data['HV_PD_data'][dose]["time"][-1] + 0.01, 1) for dose in data['HV_PD_data']}
time_vectors_SLE_PK = {dose: np.arange(-10, data['SLE_PK_data'][dose]["time"][-1] + 0.01, 1) for dose in data['SLE_PK_data']}
time_vectors_SLE_PD = {dose: np.arange(-10, data['SLE_PD_data'][dose]["time"][-1] + 0.01, 1) for dose in data['SLE_PD_data']}

all_time_vectors = {'HV': {'PK': time_vectors_PK, 'PD': time_vectors_PD},
                    'SLE': {'PK': time_vectors_SLE_PK, 'PD': time_vectors_SLE_PD}}

measurements = {'PK': 'BIIB059_mean', 'PD': 'BDCA2_median'}
ylabels = {'PK': 'Free Litifilimab Plasma Concentration [µg/ml]', 'PD': 'Total BDCA2 Expression on pDCs [% Change]'}

HV_sims = create_simulation_objects(models['HV_model'], 'HV', bodyweights['HV'], dataset=all_datasets['HV']['PK'])
SLE_sims = create_simulation_objects(models[f'SLE_model_{pDC_density}'], 'SLE', bodyweights['SLE'], dataset=all_datasets['SLE']['PK'])
simulation_objects_dict = {'HV': HV_sims, 'SLE': SLE_sims}

# Initial parameter values
merged_initial_params = [0.713, 0.0096, 2.6, 1.125, 6.987, 4.368, 2.6, 0.0055, 0.0343, 0.081, 0.95, 0.8, 0.95, 0.45, 0.2, 0.00552, 0.00552, 0.28, 5.54, 2387]
initial_params_HV, initial_params_SLE = merged_to_model_params(merged_initial_params)
all_initial_params = {'HV': initial_params_HV, 'SLE': initial_params_SLE}

for model_key in simulation_objects_dict.keys():
    print(f"Initial parameters for {model_key} model:", all_initial_params[model_key])

# The bounds for estimated parameters ensure they stay within a physiologically reasonable range during optimization
bound_factors = [1.25, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 10, 1, 1]
merged_initial_params_log = np.log(merged_initial_params)
lower_bounds_log = merged_initial_params_log - np.log(bound_factors)
upper_bounds_log = merged_initial_params_log + np.log(bound_factors)
bounds_log = Bounds(lower_bounds_log, upper_bounds_log)

print("Lower bounds:", np.exp(lower_bounds_log))
print("Upper bounds:", np.exp(upper_bounds_log))

initial_costs = evaluate_cost(all_initial_params, simulation_objects_dict, all_datasets)
total_initial_cost = 0.0

# Print all initial costs
for model_key, model_costs in initial_costs.items():
    for cost_key, cost_value in model_costs.items():
        print(f"Initial cost for {model_key} {cost_key}: {cost_value:.2f}")
    total_initial_cost += sum(model_costs.values()) 

print(f"Total initial cost: {total_initial_cost:.2f}")
    
dgf = { 'HV': {}, 'SLE': {} }
chi2_limits = { 'HV': {}, 'SLE': {} }
total_dgf = 0

# Calculate the degrees of freedom and chi2-limits for all data subsets
for model_key in simulation_objects_dict.keys():
    dataset = all_datasets[model_key]
    for data_key, current_data in dataset.items():
        dgf[model_key][data_key] = sum(np.count_nonzero(np.isfinite(current_data[dose]["SEM"])) for dose in current_data)
        chi2_limits[model_key][data_key] = chi2.ppf(0.95, dgf[model_key][data_key])
        total_dgf += dgf[model_key][data_key]
        print(f"Chi2 limit for {model_key} {data_key}: {chi2_limits[model_key][data_key]:.2f}")

total_chi2_limit = chi2.ppf(0.95, total_dgf)
print(f"Total chi2 limit: {total_chi2_limit:.2f}")

# Plot initial simulations against data for each model and simulation type
for model_key in simulation_objects_dict.keys():
    dataset = all_datasets[model_key]
    time_vectors = all_time_vectors[model_key]
    for (data_key, current_data) in dataset.items():
        time_vector = time_vectors[data_key]
        measurement = measurements[data_key]
        ylabel = ylabels[data_key]
        sims = simulation_objects_dict[model_key]
        params = all_initial_params[model_key]
        
        for dose in current_data:
            plt.figure()
            plot_sim(params, sims[dose], time_vector[dose], feature_to_plot=f"{data_key}_plasma_sim")
            plot_dataset(current_data[dose], measurement, ylabel)
            plt.title(f"{data_key} Simulation for {dose}")

plt.show()

save_dir = os.path.join(base_dir, 'Results', 'Parameter_estimation')
os.makedirs(save_dir, exist_ok=True)
best_param_path = os.path.join(save_dir, f'best_param_estimation_{pDC_density}.json')

# Load existing best parameter set if available
if os.path.exists(best_param_path) and os.path.getsize(best_param_path) > 0:
    with open(best_param_path, 'r') as f:
        best_data = json.load(f)
        best_cost = best_data['best_cost']
        best_param = np.array(best_data['best_param'])
else:
    best_cost = np.inf
    best_param = None

acceptable_params = []

# Load existing acceptable parameter sets if available
acceptable_params_path = os.path.join(save_dir, f'acceptable_params_estimation_{pDC_density}.json')
if os.path.exists(acceptable_params_path) and os.path.getsize(acceptable_params_path) > 0:
    with open(acceptable_params_path, 'r') as f:
        acceptable_params = json.load(f)

cost_function_args = (simulation_objects_dict, all_datasets)

# Cost function for the optimization which evaluates parameter sets against the chi2-limits for each data subset
# Only parameter sets that pass all chi2-limits are considered acceptable
def fcost_uncertainty(merged_params_log, simulation_objects_dict, all_datasets):
    global acceptable_params
    global best_cost
    global best_param

    merged_params = np.exp(merged_params_log)
    HV_params, SLE_params = merged_to_model_params(merged_params)
    all_params = {'HV': HV_params, 'SLE': SLE_params}

    all_costs = evaluate_cost(all_params, simulation_objects_dict, all_datasets)
    
    params_pass = True
    total_cost = 0.0

    for model_key, model_costs in all_costs.items():
        for data_key, cost in model_costs.items():
            if cost > chi2_limits[model_key][data_key]:
                params_pass = False
        total_cost += sum(model_costs.values())

    if params_pass == True:
        acceptable_params.append(merged_params.tolist())
        if total_cost < best_cost:
            best_cost = total_cost
            best_param = merged_params.copy()
            print(f"New best total cost: {best_cost:.2f}")
            for model_key, model_costs in all_costs.items():
                for data_key, cost_value in model_costs.items():
                    print(f"  {model_key} {data_key} cost: {cost_value:.2f}")

    return total_cost

# Randomly select a subset of 500 acceptable parameter sets and plot their simulations against PK and PD data
# Gives a first impression of the uncertainty in the model predictions based on the acceptable parameter sets
def plot_preliminary_uncertainty(acceptable_params, simulation_objects_dict, all_datasets, n_params_to_plot=500):
    random.shuffle(acceptable_params)

    for model_key in simulation_objects_dict.keys():
        dataset = all_datasets[model_key]
        time_vectors = all_time_vectors[model_key]
        for (data_key, current_data) in dataset.items():
            time_vector = time_vectors[data_key]
            measurement = measurements[data_key]
            ylabel = ylabels[data_key]
            sims = simulation_objects_dict[model_key]
            
            for dose in current_data:
                plt.figure()

                success_count = 0
                fail_count = 0

                for param in acceptable_params:
                    if success_count >= n_params_to_plot:
                        break
                    try:
                        HV_params, SLE_params = merged_to_model_params(param)
                        if model_key == 'HV':
                            plot_sim(HV_params, sims[dose], time_vector[dose], feature_to_plot=f"{data_key}_plasma_sim")
                        else:
                            plot_sim(SLE_params, sims[dose], time_vector[dose], feature_to_plot=f"{data_key}_plasma_sim")

                        success_count += 1
                    except RuntimeError as e:
                        if "CVODE" in str(e):
                            fail_count += 1
                            continue
                        else:
                            raise e

                plot_dataset(current_data[dose], measurement, ylabel)
                plt.title(f"{data_key} Simulations for {dose}")

                print(f"  Successful simulations for {dose}: {success_count}")
                print(f"  Failed (CVODE) simulations for {dose}: {fail_count}")

if __name__ == '__main__':
    # Perform optimization using differential evolution
    # This optimization runs multiple iterations to find the best parameters that minimize the cost function
    for i in range(5):
        res = differential_evolution(
            func=fcost_uncertainty,
            bounds=bounds_log,
            args=cost_function_args,
            x0=merged_initial_params_log,
            callback=callback_evolution_log,
            disp=True
        )

    with open(os.path.join(save_dir, f'acceptable_params_estimation_{pDC_density}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(acceptable_params)

    best_param_out = best_param.tolist() if best_param is not None else []
    with open(os.path.join(save_dir, f'best_param_estimation_{pDC_density}.json'), 'w') as f:
        json.dump({'best_cost': best_cost, 'best_param': best_param_out}, f, cls=NumpyArrayEncoder)

    print(f"Number of acceptable parameter sets collected: {len(acceptable_params)}")

    plot_preliminary_uncertainty(acceptable_params, simulation_objects_dict, all_datasets)

    plt.show()
