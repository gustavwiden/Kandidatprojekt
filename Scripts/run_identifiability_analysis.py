import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pypesto
import pypesto.sample as sample

# Global setting of skin pDC density (pDCs/mm²)
# To run the identifiability anlysis for the other pDC densities, simply change between '1', '80' and '400'
pDC_density = '80'

from utils import base_dir, load_data, load_models, create_simulation_objects, merged_to_model_params, evaluate_cost

save_dir_MCMC = os.path.join(base_dir, 'Results', 'MCMC')
os.makedirs(save_dir_MCMC, exist_ok=True)

save_dir_PL = os.path.join(base_dir, 'Results', 'Profile_likelihood')
os.makedirs(save_dir_PL, exist_ok=True)

data = load_data('HV_PK_data', 'HV_PD_data', 'SLE_PK_data', 'SLE_PD_data')

models = load_models('HV_model', f'SLE_model_{pDC_density}')

all_datasets = {'HV': {'PK': data['HV_PK_data'], 'PD': data['HV_PD_data']}, 
                'SLE': {'PK': data['SLE_PK_data'], 'PD': data['SLE_PD_data']}}

# Average bodyweight (kg) for HV and SLE patients (cohort 1-7 and cohort 8 respectively in the phase 1 trial)
bodyweights = {'HV': 73, 'SLE': 69}

HV_sims = create_simulation_objects(models['HV_model'], 'HV', bodyweights['HV'], dataset=all_datasets['HV']['PK'])
SLE_sims = create_simulation_objects(models[f'SLE_model_{pDC_density}'], 'SLE', bodyweights['SLE'], dataset=all_datasets['SLE']['PK'])
simulation_objects_dict = {'HV': HV_sims, 'SLE': SLE_sims}

# Initial parameter values
merged_initial_params = [0.713, 0.0096, 2.6, 1.125, 6.987, 4.368, 2.6, 0.0055, 0.0343, 0.081, 0.95, 0.8, 0.95, 0.45, 0.2, 0.00552, 0.00552, 0.28, 5.54, 2387]
initial_params_HV, initial_params_SLE = merged_to_model_params(merged_initial_params)
all_initial_params = {'HV': initial_params_HV, 'SLE': initial_params_SLE}

for model_key in simulation_objects_dict.keys():
    print(f"Initial parameters for {model_key} model:", all_initial_params[model_key])

# The bounds for estimated parameters are set to be significantly wider for identifiability analysis compared to parameter estimation
# This is to ensure that the MCMC and profile likelihood analyses are not restricted from exploring the parameter space
bound_factors = [5, 15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 5, 5, 20, 1, 1]
merged_initial_params_log = np.log(merged_initial_params)
lower_bounds = merged_initial_params_log - np.log(bound_factors)
upper_bounds = merged_initial_params_log + np.log(bound_factors)
bounds_log = (lower_bounds, upper_bounds)

print("Lower bounds:", np.exp(lower_bounds))
print("Upper bounds:", np.exp(upper_bounds))

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

# Dictionary with names and indices for the estimated parameters
parameters_dict = {'F': 0, 'ka': 1, 'RC2': 13, 'CL_HV': 15, 'CL_SLE': 16, 'kdeg': 17}

best_result_dir = os.path.join(base_dir, 'Results', 'Parameter_estimation')
best_result_file = os.path.join(best_result_dir, f'best_param_estimation_{pDC_density}.json')

if not os.path.exists(best_result_file) or os.path.getsize(best_result_file) == 0:
    raise RuntimeError(f'Best result file not found or empty: {best_result_file}; run parameter estimation first')
with open(best_result_file, 'r') as f:
    best_data_json = json.load(f)
    best_param = np.array(best_data_json['best_param'])

# Insert the selected parameters back into the full parameter vector
def insert_params(selected_params, best_param_full, param_indices):
    full = best_param_full.copy()
    full[param_indices] = selected_params
    return full

# Function to evaluate the cost for MCMC sampling
def fcost_sampling(params_reduced, simulation_objects_dict_local, all_datasets_local, SaveParams=True, sampling_params=None):
    param_indices = list(parameters_dict.values())
    full = insert_params(params_reduced, best_param.copy(), param_indices)
    
    HV_params, SLE_params = merged_to_model_params(full)
    all_params = {'HV': HV_params, 'SLE': SLE_params}
    
    pass_check = True
    total_cost = 0.0

    try:
        all_costs = evaluate_cost(all_params, simulation_objects_dict_local, all_datasets_local)
        for model_key, model_costs in all_costs.items():
            for data_key, cost in model_costs.items():
                if cost > chi2_limits[model_key][data_key]:
                    pass_check = False
            total_cost += sum(model_costs.values())
    except Exception:
        pass_check = False
        total_cost += 1e30

    if SaveParams and pass_check and sampling_params is not None:
        sampling_params.append(full.tolist())
        
    return total_cost

# Generates the plot for Supplementary Figure 10
def plot_mcmc(trace_array):
    param_names = list(parameters_dict.keys())
    
    ci_bounds = {}
    ci_path = os.path.join(save_dir_PL, f'95_CI_bounds_1_dgf_{pDC_density}.txt')
    
    if os.path.exists(ci_path):
        with open(ci_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith('95%') or line.startswith('-'):
                    continue
                if ':' in line:
                    name, value_text = line.split(':', 1)
                    name = name.strip()
                    if name in param_names:
                        value_text = value_text.strip()
                        values = value_text.strip('[]').split(',')
                        if len(values) == 2:
                            ci_bounds[name] = (float(values[0].strip()), float(values[1].strip()))
                            
    rows, cols = 2, 3
    fig, axs = plt.subplots(rows, cols, figsize=(14, 7))
    axes = axs.flatten()
    num_params = trace_array.shape[1]
    color = plt.cm.Blues(np.linspace(0.85, 0.85, 1))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.3, hspace=0.35)

    for i in range(num_params):
        ax = axes[i]
        data_for_hist = trace_array[:, i]
        ax.hist(data_for_hist, bins='auto', color=color[0])
        current_param_name = param_names[i]

        if current_param_name in ci_bounds:
            lower, upper = ci_bounds[current_param_name]
            ax.axvline(lower, linestyle=':', color='black', linewidth=1.5, alpha=0.7)
            ax.axvline(upper, linestyle=':', color='black', linewidth=1.5, alpha=0.7)

            span = upper - lower
            pad = 0.25 * span
            ax.set_xlim(lower - pad, upper + pad)

        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_xlabel(f'{current_param_name} Value', fontsize=14)

        if current_param_name in ci_bounds:
            ax.text(0.5, 0.92, '95% Confidence Interval', transform=ax.transAxes, ha='center', va='center', fontsize=10, color='black')
            ax.annotate('', xy=(0.82, 0.925), xytext=(0.76, 0.925), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle='->', color='black', lw=1.2, shrinkA=0, shrinkB=0))
            ax.annotate('', xy=(0.18, 0.925), xytext=(0.24, 0.925), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle='->', color='black', lw=1.2, shrinkA=0, shrinkB=0))

        formatter = ticker.FuncFormatter(lambda x, _: f"{x:.2e}")
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

        # Appropriate y-axis when using 1e6 samples
        ax.set_ylim(0, 13000)
        ax.set_yticks([0, 2000, 4000, 6000, 8000, 10000, 12000])

    for j in range(num_params, len(axes)):
        fig.delaxes(axes[j])

    save_path = os.path.join(save_dir_MCMC, "MCMC_mPBPK-model_model.svg")
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()

# Run MCMC sampling
def run_mcmc():
    csv_path = os.path.join(save_dir_MCMC, 'MCMC_sampling_results.csv')
    
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        raise FileExistsError(
            f"MCMC results already exist at {csv_path}. "
            "Please rename or move existing file to prevent accidental overwriting."
        )

    sampling_params = []
    param_indices = list(parameters_dict.values())
    param_names = list(parameters_dict.keys())

    lb = np.exp(lower_bounds)[param_indices]
    ub = np.exp(upper_bounds)[param_indices]

    # Proxy for pypesto (expects params in linear space)
    def proxy_f(params):
        return fcost_sampling(params, simulation_objects_dict, all_datasets, True, sampling_params)

    best_selected_params = best_param[param_indices]
    parameter_scales = ['lin'] * len(best_selected_params)
    
    custom_objective = pypesto.Objective(fun=proxy_f, grad=None)
    custom_problem = pypesto.Problem(objective=custom_objective, lb=lb, ub=ub, x_guesses=[best_selected_params], x_scales=parameter_scales, x_names=param_names)
    
    n_samples = int(1e6)
    sampler = sample.AdaptiveMetropolisSampler()

    # Note that running the MCMC sampling with 1e6 samples will take several hours
    print("Starting MCMC sampling...")
    result_sampling = sample.sample(problem=custom_problem, n_samples=n_samples, sampler=sampler, result=None, x0=best_selected_params)

    trace = np.array(result_sampling.sample_result['trace_x'])[0]
    np.savetxt(csv_path, trace, delimiter=',')
    print(f"MCMC sampling complete. Results saved to {csv_path}")

    plot_mcmc(trace)
    
    return trace

# Cost function for profile likelihood
def fcost_PL(param_log, param_index, PL_revValue):
    params = np.exp(param_log)
    HV_params, SLE_params = merged_to_model_params(params)
    
    try:
        all_costs = evaluate_cost({'HV': HV_params, 'SLE': SLE_params}, simulation_objects_dict, all_datasets)
        joint_cost = sum(sum(model_costs.values()) for model_costs in all_costs.values())
    except Exception:
        joint_cost = 1e30

    penalty = 1e6 * (param_log[param_index] - PL_revValue) ** 2
    return joint_cost + penalty

# Generates the plots for Supplementary Figures 7-9 when run with mode=1_dgf
def plot_profile_likelihood(mode):
    csv_path = os.path.join(save_dir_PL, f"PL_plot_data_{mode}_{pDC_density}.csv")
    
    if not os.path.exists(csv_path):
        print(f"PL data not found at {csv_path}.")
        return

    df = pd.read_csv(csv_path)
    
    for param_name in df['Parameter_Name'].unique():
        param_df = df[df['Parameter_Name'] == param_name].copy()
        param_df.sort_values(by='Parameter_Value', inplace=True)
        
        x_vals = param_df['Parameter_Value'].tolist()
        y_vals = param_df['Total_Cost'].tolist()
        
        valid_mask = param_df['Is_Valid'] == True
        x_valid = param_df[valid_mask]['Parameter_Value'].tolist()
        y_valid = param_df[valid_mask]['Total_Cost'].tolist()
        
        x_invalid = param_df[~valid_mask]['Parameter_Value'].tolist()
        y_invalid = param_df[~valid_mask]['Total_Cost'].tolist()
        
        limit_val = param_df['Limit'].iloc[0]
        initial_cost = min(y_vals)

        plt.figure()
        plt.plot(x_vals, y_vals, 'k--', alpha=0.5, label='PL Profile')
        
        if mode == '1_dgf':
            valid_label = 'Inside 95% CI (1 dgf)'
            invalid_label = 'Outside 95% CI'
            line_label = '95% CI Limit'
            y_max = initial_cost + 10.0
        else:
            valid_label = 'Accepted (Passed All Limits)'
            invalid_label = 'Rejected (Failed Individual)'
            line_label = 'Total Chi² Limit'
            y_max = limit_val + 10.0

        if x_valid:
            plt.scatter(x_valid, y_valid, color='black', zorder=5, label=valid_label)
        if x_invalid:
            plt.scatter(x_invalid, y_invalid, color='red', marker='x', zorder=5, label=invalid_label)

        plt.axhline(y=limit_val, linestyle='--', color='r', label=line_label)
        plt.xlabel(f'{param_name} Value')
        plt.ylabel('Total Cost')
        plt.ylim(initial_cost - 1.0, y_max)
        plt.legend(fontsize='small', loc='upper center')
        
        save_path = os.path.join(save_dir_PL, f"PL_{mode}_{param_name}_{pDC_density}.svg")
        plt.savefig(save_path, format='svg')
        plt.close()

# Run profile likelihood - Pass the degrees of freedom as the argument (1_dgf or N_dgf)
# 1_dgf is used to calculate 95% confidence intervals
# N_dgf is used to gather all acceptable parameter sets for simulation uncertainty bands
def run_profile_likelihood(mode='1_dgf'):
    bounds_log_pairs = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    best_param_log = np.log(best_param)
    
    # Assign step sizes based on the mode
    if mode == '1_dgf':
        step_sizes = {'F': 0.01, 'ka': 0.025, 'RC2': 0.005, 'CL_HV': 0.005, 'CL_SLE': 0.01, 'kdeg': 0.01}
    else:
        step_sizes = {'F': 0.04, 'ka': 0.1, 'RC2': 0.04, 'CL_HV': 0.02, 'CL_SLE': 0.04, 'kdeg': 0.04}
    
    # Check if a parameter set is accepted
    def check_validity(params, limit_1_dgf):
        try:
            HV_params, SLE_params = merged_to_model_params(params)
            all_costs = evaluate_cost({'HV': HV_params, 'SLE': SLE_params}, simulation_objects_dict, all_datasets)
            total_cost = sum(sum(model_costs.values()) for model_costs in all_costs.values())

            # Total cost must not the best found cost from parameter estimation + 3.841
            if mode == '1_dgf':
                return (total_cost <= limit_1_dgf), total_cost

            # All costs must pass the respective chi2-limit for each data subset
            elif mode == 'N_dgf':
                for model_key, model_costs in all_costs.items():
                    for data_key, cost in model_costs.items():
                        if cost > chi2_limits[model_key][data_key]:
                            return False, total_cost
                return True, total_cost
                
        except Exception:
            return False, 1e30

    all_plot_data = []
    master_acceptable_params = []

    # For each selected parameter, perform the PL scan
    for param_name, param_idx in parameters_dict.items():
        step_size = step_sizes[param_name]
        parameterIdx = param_idx
        nSteps = 25 
        start_params = np.exp(best_param_log)
        
        _, initial_cost = check_validity(start_params, np.inf)
        limit_1_dgf = initial_cost + 3.841

        plot_limit = limit_1_dgf if mode == '1_dgf' else total_chi2_limit

        start_valid, _ = check_validity(start_params, limit_1_dgf)

        all_plot_data.append([param_name, start_params[parameterIdx], initial_cost, start_valid, plot_limit])
        
        PL_params_to_save = []
        if start_valid: 
            PL_params_to_save.append(start_params.tolist())

        print(f"--- Scanning {param_name} ---")

        for direction in [-1, 1]:
            x_opt_prev = best_param_log.copy()
            for step in range(1, nSteps + 1): 
                PL_revValue = best_param_log[parameterIdx] + direction * step_size * step
                x0 = x_opt_prev.copy()
                x0[parameterIdx] = PL_revValue 

                # Begin with Fast Gradient descent (L-BFGS-B)
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

                # Fallback: Nelder-Mead
                cost_jump = res.fun - initial_cost if success else np.inf
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
                
                is_valid, pure_cost = check_validity(step_params, limit_1_dgf)

                all_plot_data.append([param_name, np.exp(PL_revValue), pure_cost, is_valid, plot_limit])
                print(f"  Step {step} dir {direction}: pure cost={pure_cost:.3f}, valid={is_valid}")

                if is_valid:
                    PL_params_to_save.append(step_params.tolist())

                if mode == '1_dgf' and pure_cost > (initial_cost + 10.0):
                    break
                elif mode == 'N_dgf' and pure_cost > (total_chi2_limit + 10.0):
                    break

        master_acceptable_params.extend(PL_params_to_save)

        save_csv = os.path.join(save_dir_PL, f"acceptable_params_PL_{mode}_{param_name}_{pDC_density}.csv")
        with open(save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(PL_params_to_save)
    
    for param_name, param_idx in parameters_dict.items():
        save_txt = os.path.join(save_dir_PL, f"95_CI_bounds_{mode}_{pDC_density}.txt")
        with open(save_txt, 'w') as f_out:
            f_out.write(f"95% Confidence Intervals from Profile Likelihood ({mode})\n")
            f_out.write("-" * 50 + "\n")
            
            for param_name, param_idx in parameters_dict.items():
                csv_path = os.path.join(save_dir_PL, f"acceptable_params_PL_{mode}_{param_name}_{pDC_density}.csv")
                
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path, header=None)
                        lower_bound = df[param_idx].min()
                        upper_bound = df[param_idx].max()
                        f_out.write(f"{param_name}:\t[{lower_bound:.4f}, {upper_bound:.4f}]\n")
                    except Exception as e:
                        f_out.write(f"{param_name}:\tError processing file -> {e}\n")
                else:
                    f_out.write(f"{param_name}:\tCSV not found\n")

    plot_data_csv = os.path.join(save_dir_PL, f"PL_plot_data_{mode}_{pDC_density}.csv")
    with open(plot_data_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter_Name', 'Parameter_Value', 'Total_Cost', 'Is_Valid', 'Limit'])
        writer.writerows(all_plot_data)

    master_csv = os.path.join(save_dir_PL, f"acceptable_params_PL_{mode}_{pDC_density}.csv")
    with open(master_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(master_acceptable_params)
        
    print(f"Profile likelihood analysis complete. Generating PL plots for mode {mode}...")
    plot_profile_likelihood(mode)


# Run profile likelihood first so the 95% confidence intervals (CI) txt-files are created
# The plot_mcmc function requires these files to plot upper/lower CI bounds as in Supplementary Figure 10
if __name__ == '__main__':
    # run_profile_likelihood(mode='N_dgf')
    # run_mcmc()
    pass