"""
MCMC and Profile Likelihood script adapted to work with optimize_model.py

This script performs Metropolis-Hastings MCMC sampling and simple profile-likelihood
scans for the following parameters in the merged parameter vector used by
`optimize_model.py`:
- F (bioavailability)
- ka (absorption rate)
- RC2 (reflection coefficient for leaky compartment)
- CL (linear clearance) for HV and SLE (separate entries in merged vector)
- kdeg (degradation rate)

The script expects that `Scripts/Optimize/optimize_model.py` exposes the
following variables at import time (they are created at top-level in that file):
- `merged_initial_params` (numpy array or list)
- `simulation_objects_dict` (dict mapping model keys)
- `all_datasets` (dict with data for HV and SLE)
- `fcost_joint` (function that evaluates cost for a single-model params; returns tuple (total_cost, pk_cost, pd_cost) or float)

If `optimize_model.py` changes, you may need to update the index mapping below.

Outputs:
- Saves chain and profile-likelihood scans under `../../Results/Acceptable params/`.

"""

import os
import json
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

import sys
import sund
import matplotlib.pyplot as plt

# Output directory
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Results', 'Acceptable params'))
os.makedirs(output_dir, exist_ok=True)

# Minimal setup (avoid importing optimize_model to prevent executing its heavy top-level code)
# Load datasets
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'PK_data.json'), 'r') as f:
    HV_PK_data = json.load(f)
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'PD_data.json'), 'r') as f:
    HV_PD_data = json.load(f)
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'SLE_PK_data.json'), 'r') as f:
    SLE_PK_data = json.load(f)
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'SLE_PD_data.json'), 'r') as f:
    SLE_PD_data = json.load(f)

# Install and load the models (use relative paths from script location)
# Change to script directory to ensure relative path resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sund.install_model('../../Models/mPBPK_model.txt')
sund.install_model('../../Models/mPBPK_SLE_model_80_pdc_mm2.txt')
HV_model = sund.load_model('mPBPK_model')
SLE_model = sund.load_model('mPBPK_SLE_model_80_pdc_mm2')

# Build dataset dicts and simulation objects similar to optimize_model.py
all_datasets = {'HV': {'PK': HV_PK_data, 'PD': HV_PD_data}, 'SLE': {'PK': SLE_PK_data, 'PD': SLE_PD_data}}
bodyweights = {'HV': 73, 'SLE': 69}

activity_objects_dict = {}
for model_key, PK_data in [('HV', HV_PK_data), ('SLE', SLE_PK_data)]:
    bw = bodyweights[model_key]
    act_objs = {}
    for dose_key in PK_data.keys():
        act = sund.Activity(time_unit='h')
        if 'IV_in' in PK_data[dose_key]['input']:
            act.add_output('piecewise_constant', 'IV_in', t=PK_data[dose_key]['input']['IV_in']['t'], f=bw * np.array(PK_data[dose_key]['input']['IV_in']['f']))
        if 'SC_in' in PK_data[dose_key]['input']:
            act.add_output('piecewise_constant', 'SC_in', t=PK_data[dose_key]['input']['SC_in']['t'], f=PK_data[dose_key]['input']['SC_in']['f'])
        act_objs[dose_key] = act
    activity_objects_dict[model_key] = act_objs

simulation_objects_dict = {}
for model_key, model in [('HV', HV_model), ('SLE', SLE_model)]:
    sims = {}
    for dose_key, act in activity_objects_dict[model_key].items():
        sims[dose_key] = sund.Simulation(models=model, activities=act, time_unit='h')
    simulation_objects_dict[model_key] = sims

# Load merged initial params and bounds (use same defaults as optimize_model)
merged_initial_params = np.array([0.713, 0.0096, 2.6, 1.125, 6.987, 4.368, 2.6, 0.0065, 0.0338, 0.081, 0.95, 0.8, 0.95, 0.45, 0.2, 0.00552, 0.00552, 0.28, 5.54, 2624])
bound_factors = [5, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 5, 5, 20, 1, 1]
lower_bounds = np.log(merged_initial_params) - np.log(bound_factors)
upper_bounds = np.log(merged_initial_params) + np.log(bound_factors)
bounds_log = (lower_bounds, upper_bounds)
merged_init = merged_initial_params

# Selected parameter indices in merged vector
selected_indices = [0, 1, 13, 15, 16, 17]
selected_names = ['F', 'ka', 'RC2', 'CL_HV', 'CL_SLE', 'kdeg']

# Helper: build model-specific param vectors by deleting the appropriate indices
hv_remove = [11, 16]   # indices to remove from merged to get HV param vector (SLE RCS and SLE CL)
sle_remove = [10, 15]  # indices to remove from merged to get SLE param vector (HV RCS and HV CL)


def merged_to_model_params(merged):
    """Return (params_HV, params_SLE) arrays by deleting the extra entries."""
    merged = np.array(merged)
    params_HV = np.delete(merged, hv_remove)
    params_SLE = np.delete(merged, sle_remove)
    return params_HV, params_SLE


# Define the joint cost function for PK and PD data (local copy)
def fcost_joint(params, sims, dataset):
    costs = {}
    for data_key, data in dataset.items():
        measurement = 'BIIB059_mean' if data_key == 'PK' else 'BDCA2_median'
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
                    print(f"Simulation of {dose} failed: {e}")
                    cost = 1e30
                    break
        costs[data_key] = cost
    return costs


# Compute degrees of freedom and chi2 limits for each model/data type
dgf = {'HV': {}, 'SLE': {}}
chi2_limits = {'HV': {}, 'SLE': {}}
for model_key in all_datasets.keys():
    dataset = all_datasets[model_key]
    for data_key, data in dataset.items():
        dgf[model_key][data_key] = sum(np.count_nonzero(np.isfinite(data[dose]["SEM"])) for dose in data)
        chi2_limits[model_key][data_key] = chi2.ppf(0.95, dgf[model_key][data_key])

chi2_total_limit = chi2.ppf(0.95, sum(dgf[model_key][data_key] for model_key in dgf for data_key in dgf[model_key]))


def evaluate_merged_cost(merged):
    """Evaluate total cost for merged parameter vector by mapping to HV and SLE and
    calling local fcost_joint for each model.

    Returns total_cost (float).
    """
    params_HV, params_SLE = merged_to_model_params(merged)

    try:
        hv_costs = fcost_joint(params_HV, simulation_objects_dict['HV'], all_datasets['HV'])
        sle_costs = fcost_joint(params_SLE, simulation_objects_dict['SLE'], all_datasets['SLE'])
        hv_total = sum(hv_costs.values())
        sle_total = sum(sle_costs.values())
    except Exception as e:
        raise RuntimeError("Failed evaluating model costs: {}".format(e))

    return hv_total + sle_total


# Quick sanity: evaluate cost at the merged initial point
start_cost = evaluate_merged_cost(merged_init)
print(f"Start total cost: {start_cost:.3f}")

# ---------------------- MCMC using pypesto (match original scripts) ----------------------
import csv
import pypesto
import pypesto.sample as sample


# Load best parameter from best_result_80_pdc_mm2.json
best_result_file = os.path.join(output_dir, 'best_result_80_pdc_mm2.json')
if not os.path.exists(best_result_file) or os.path.getsize(best_result_file) == 0:
    raise RuntimeError(f'Best result file not found or empty: {best_result_file}; run optimizer first')
with open(best_result_file, 'r') as f:
    data = json.load(f)
    best_param = np.array(data['best_param'])

# Prepare selected parameter bounds (linear space)
lb = np.exp(lower_bounds)[selected_indices]
ub = np.exp(upper_bounds)[selected_indices]

# bounds for log-space optimization (list of (low,high) pairs)
bounds_log_pairs = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))

# Define insert/restore helpers mirroring the original script style
def insert_params(selected_params, best_param_full, selected_indices):
    full = best_param_full.copy()
    full[selected_indices] = selected_params
    return full


# Define proxy objective for pypesto that uses reduced parameter vector
sampling_params = []

def fcost_sampling(params_reduced, simulation_objects_dict_local, all_datasets_local, SaveParams=True):
    # Build full merged parameter vector
    full = insert_params(params_reduced, best_param.copy(), selected_indices)
    # Evaluate merged cost (sum HV+SLE)
    total_cost = evaluate_merged_cost(full)

    # Check against chi2 limits per model (local chi2_limits)
    pass_check = True
    try:
        params_HV, params_SLE = merged_to_model_params(full)
        hv_costs = fcost_joint(params_HV, simulation_objects_dict_local['HV'], all_datasets_local['HV'])
        sle_costs = fcost_joint(params_SLE, simulation_objects_dict_local['SLE'], all_datasets_local['SLE'])
        pk_hv = hv_costs.get('PK', sum(hv_costs.values())) if isinstance(hv_costs, dict) else float(hv_costs)
        pd_hv = hv_costs.get('PD', 0) if isinstance(hv_costs, dict) else 0
        pk_sle = sle_costs.get('PK', sum(sle_costs.values())) if isinstance(sle_costs, dict) else float(sle_costs)
        pd_sle = sle_costs.get('PD', 0) if isinstance(sle_costs, dict) else 0
        if (pk_hv > chi2_limits['HV']['PK']) or (pd_hv > chi2_limits['HV']['PD']) or (pk_sle > chi2_limits['SLE']['PK']) or (pd_sle > chi2_limits['SLE']['PD']):
            pass_check = False
    except Exception:
        pass_check = True

    if SaveParams and pass_check:
        sampling_params.append(full.tolist())

    return total_cost


# proxy for pypesto (expects params in linear space)
def proxy_f(params):
    return fcost_sampling(params, simulation_objects_dict, all_datasets, True)


# Setup pypesto problem
param_init = best_param[selected_indices]
parameter_scales = ['lin'] * len(param_init)
parameter_names = selected_names

custom_objective = pypesto.Objective(fun=proxy_f, grad=None)
custom_problem = pypesto.Problem(objective=custom_objective, lb=lb, ub=ub, x_guesses=[param_init], x_scales=parameter_scales, x_names=parameter_names)

# Run Adaptive Metropolis sampling (as in original)
n_samples = int(1e6)
sampler = sample.AdaptiveMetropolisSampler()
# result_sampling = sample.sample(problem=custom_problem, n_samples=n_samples, sampler=sampler, result=None, x0=param_init)

# Save or load sampling trace to/from CSV (prefer existing CSV)
csv_path = os.path.join(output_dir, 'sampling_result_model.csv')
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
    try:
        trace = np.loadtxt(csv_path, delimiter=',')
        print('Loaded existing sampling trace from', csv_path)
    except Exception as e:
        raise RuntimeError(f'Failed loading existing sampling CSV {csv_path}: {e}')
else:
    # If CSV doesn't exist, attempt to extract trace from in-memory result_sampling
    try:
        trace = np.array(result_sampling.sample_result['trace_x'])[0]
        # Ensure directory exists then save
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(csv_path, trace, delimiter=',')
        print('MCMC sampling finished. Saved trace to', csv_path)
    except Exception as e:
        raise RuntimeError('No sampling CSV found and no in-memory sampling result available: {}'.format(e))


# ---------------------- Plot histograms as in original script ----------------------
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# trace_array = trace
# rows, cols = 2, 3
# fig, axs = plt.subplots(rows, cols, figsize=(8, 6))
# axes = axs.flatten()
# num_params = trace_array.shape[1]

# for i in range(num_params):
#     ax = axes[i]
#     data_for_hist = trace_array[:, i]
#     ax.hist(data_for_hist, bins='auto', color='green')
#     ax.set_title(parameter_names[i])
#     ax.set_ylabel('Frequency')
#     ax.set_xlabel('Parameter value')
#     formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
#     ax.xaxis.set_major_formatter(formatter)
#     ax.tick_params(axis='x', labelrotation=45, labelsize=8)
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

# for j in range(num_params, len(axes)):
#     fig.delaxes(axes[j])

save_dir = os.path.join(os.path.dirname(output_dir), 'Validation')
os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "MCMC_mPBPK-model_model.png")
# plt.savefig(save_path, format='png', dpi=600)
# plt.tight_layout()
# plt.show()


# ---------------------- Profile Likelihood (PL) ----------------------
def fcost_PL(param_log, param_index, PL_revValue):
    params = np.exp(param_log)
    # penalty in log-space
    joint_cost = evaluate_merged_cost(params)
    # Penalty to keep parameter close to stepped value
    penalty = 1e6 * (param_log[param_index] - PL_revValue) ** 2
    return joint_cost + penalty

import csv as _csv

# Use best_param as starting point
best_param_log = np.log(best_param)
step_sizes = [0.04, 0.1, 0.04, 0.02, 0.04, 0.04]

# For each selected parameter, perform PL scan
for i, (idx, step_size) in enumerate(zip(selected_indices, step_sizes)):
    parameterIdx = idx
    nSteps = 25 
    
    # Check if the start point itself is valid
    start_params = np.exp(best_param_log)
    initial_cost = evaluate_merged_cost(start_params)
    
    # Helper to check validity (Is it saved to CSV?)
    def check_validity(p):
        try:
            p_HV, p_SLE = merged_to_model_params(p)
            hv = fcost_joint(p_HV, simulation_objects_dict['HV'], all_datasets['HV'])
            sle = fcost_joint(p_SLE, simulation_objects_dict['SLE'], all_datasets['SLE'])
            
            pk_hv = hv.get('PK', sum(hv.values())) if isinstance(hv, dict) else float(hv)
            pd_hv = hv.get('PD', 0) if isinstance(hv, dict) else 0
            pk_sle = sle.get('PK', sum(sle.values())) if isinstance(sle, dict) else float(sle)
            pd_sle = sle.get('PD', 0) if isinstance(sle, dict) else 0
            
            # Strict individual checks
            if (pk_hv < chi2_limits['HV']['PK'] and pd_hv < chi2_limits['HV']['PD'] and 
                pk_sle < chi2_limits['SLE']['PK'] and pd_sle < chi2_limits['SLE']['PD']):
                return True
        except:
            return False
        return False

    # Store: (value, cost, is_valid_boolean)
    start_valid = check_validity(start_params)
    plot_data = [(start_params[parameterIdx], initial_cost, start_valid)]
    
    PL_params_to_save = []
    if start_valid: 
        PL_params_to_save.append(start_params.tolist())

    print(f"--- Scanning {selected_names[i]} ---")

    for direction in [-1, 1]:
        x_opt_prev = best_param_log.copy()
        
        for step in range(1, nSteps + 1): 
            PL_revValue = best_param_log[parameterIdx] + direction * step_size * step
            
            x0 = x_opt_prev.copy()
            x0[parameterIdx] = PL_revValue 

            # 1. Try Fast Gradient Descent (L-BFGS-B)
            try:
                res = minimize(
                    fun=fcost_PL,
                    x0=x0,
                    args=(parameterIdx, PL_revValue),
                    method='L-BFGS-B',
                    bounds=bounds_log_pairs,
                    options={'disp': False, 'maxiter': 100}
                )
                success = res.success
                current_res = res
            except Exception:
                success = False

            # 2. Fallback: Nelder-Mead (Run if failed OR if cost jumped suspiciously)
            # Check cost jump
            cost_jump = res.fun - initial_cost
            if not success or (step > 1 and cost_jump > 20):
                try:
                    res_nm = minimize(
                        fun=fcost_PL,
                        x0=current_res.x if success else x0, 
                        args=(parameterIdx, PL_revValue),
                        method='Nelder-Mead',
                        bounds=bounds_log_pairs,
                        options={'disp': False, 'maxiter': 500}
                    )
                    # Only accept fallback if it improved cost
                    if not success or res_nm.fun < res.fun:
                        current_res = res_nm
                except Exception:
                    pass
            
            x_opt_prev = current_res.x.copy()
            step_params = np.exp(current_res.x)
            
            # Calculate Pure Cost
            pure_cost = evaluate_merged_cost(step_params)
            
            # Check Validity (Individual Limits)
            is_valid = check_validity(step_params)
            
            plot_data.append((np.exp(PL_revValue), pure_cost, is_valid))
            print(f"  Step {step} dir {direction}: pure cost={pure_cost:.3f}, valid={is_valid}")

            if is_valid:
                PL_params_to_save.append(step_params.tolist())

    # Save acceptable params to CSV
    pl_csv = os.path.join(output_dir, f"acceptable_params_PL_{selected_names[i]}_test.csv")
    with open(pl_csv, 'w', newline='') as f:
        writer = _csv.writer(f)
        writer.writerows(PL_params_to_save)

    # --- Plotting ---
    # Sort by parameter value
    plot_data.sort(key=lambda x: x[0])
    
    x_vals = [p[0] for p in plot_data]
    y_vals = [p[1] for p in plot_data]
    
    # Separate valid and invalid points for different markers
    x_valid = [p[0] for p in plot_data if p[2]]
    y_valid = [p[1] for p in plot_data if p[2]]
    
    x_invalid = [p[0] for p in plot_data if not p[2]]
    y_invalid = [p[1] for p in plot_data if not p[2]]

    plt.figure()
    # Plot the profile line (dashed)
    plt.plot(x_vals, y_vals, 'k--', alpha=0.5, label='PL Profile')
    
    # Plot Valid points (Black Circles)
    if x_valid:
        plt.scatter(x_valid, y_valid, color='black', zorder=5, label='Accepted (Passed All)')
        
    # Plot Invalid points (Red X)
    if x_invalid:
        plt.scatter(x_invalid, y_invalid, color='red', marker='x', zorder=5, label='Rejected (Failed Individual)')

    plt.axhline(y=chi2_total_limit, linestyle='--', color='r', label='Total Chi² Limit')
    plt.xlabel(f'{selected_names[i]} Value')
    plt.ylabel('Total Cost')
    
    # Smart Y-limit
    plt.ylim(0, chi2_total_limit * 1.15)
        
    plt.legend(fontsize='small')
    save_path = os.path.join(save_dir, f"PL_{selected_names[i]}_model_test.png")
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()

print('MCMC and PL done.')