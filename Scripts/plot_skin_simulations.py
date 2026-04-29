# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

from utils import base_dir, load_data, load_models, load_params, create_simulation_objects, calculate_uncertainty, simulate, save_plot, get_response_time, check_suppression_maintained

data = load_data('SLE_CLE_PK_validation_data', 'SLE_IV_dose_skin_PD_response_data', 'SLE_SC_dose_skin_PD_response_data')  

models = load_models('HV_model', 'SLE_model_1', 'SLE_model_10', 'SLE_model_80', 'SLE_model_400', 'HV_model_high')    

params = load_params('final_params', 'acceptable_params_1', 'acceptable_params_10', 'acceptable_params_80', 'acceptable_params_400')

# Overall average bodyweight for healthy volunteers (cohort 1-7) and SLE patients (cohort 8) in the phase 1 trial
bodyweight = 73

# Define the time vectors for each dose
# time_vector_IV = np.arange(-10, 5100, 1)
# time_vector_SC = np.arange(-10, 6800, 1)
# time_vector_AUC = np.arange(0, 2688, 1)
# time_vector_ratio = np.arange(0, 8000, 1)
# time_vectors_IV_SC = {'IV': time_vector_IV, 'SC': time_vector_SC}

# Conversion factor for plotting time in weeks (keep simulations in hours)
time_vectors = {'IV': np.arange(-10, 5100, 1), 'SC': np.arange(-10, 7500, 1), 'AUC': np.arange(0, 2688, 1), 'Ratio': np.arange(0, 8000, 1)}

# Define a function to plot PK and PD simulations in skin together
# Change BDCA2 baseline concentration in mPBPK_SLE_model.txt to create plots for different pDC densities in skin lesions
def plot_skin_PK_PD_simulations_together(params, models, time_vector):

    patient_labels = ['1', '10', '80', '400']
    doses = [0.05, 0.3, 1, 3, 10, 20, 40, 60]
    PK_colors = plt.cm.Blues(np.linspace(0.6, 0.9, 4))
    PD_colors = plt.cm.Reds(np.linspace(0.6, 0.9, 4))
    
    response_results = {label: {"Dose": doses, "Best": [], "Fast": [], "Slow": []} for label in patient_labels}
    time_weeks = time_vector / 168

    for i, label in enumerate(patient_labels):
        save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PKPD', label)
        model = models[f'SLE_model_{label}']
        acceptable_params = params[f'acceptable_params_{label}']
        final_params = params[f'final_params'][f'SLE_{label}']

        for dose in doses:
            sim = create_simulation_objects(model, 'SLE', bodyweight, custom_IV_doses=[dose])

            y_pk_min, y_pk_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PK_skin_sim')
            y_pd_min, y_pd_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PD_skin_sim')
            y_pk_best = simulate(final_params, sim, time_vector, 'PK_skin_sim')
            y_pd_best = simulate(final_params, sim, time_vector, 'PD_skin_sim')

            response_threshold = -90
            startpoint = 0

            response_results[label]["Best"].append(get_response_time(y_pd_best, response_threshold, startpoint, time_weeks, 'PD'))
            response_results[label]["Fast"].append(get_response_time(y_pd_max, response_threshold, startpoint, time_weeks, 'PD'))
            response_results[label]["Slow"].append(get_response_time(y_pd_min, response_threshold, startpoint, time_weeks, 'PD'))

            fig, ax1 = plt.subplots(figsize=(10, 8))
            ax2 = ax1.twinx()

            ax1.fill_between(time_weeks, y_pk_min, y_pk_max, color=PK_colors[i], alpha=0.5, label='PK Uncertainty')
            ax2.fill_between(time_weeks, y_pd_min, y_pd_max, color=PD_colors[i], alpha=0.5, label='PD Uncertainty')        
            ax1.plot(time_weeks, y_pk_best, color=PK_colors[i], label='PK simulation')
            ax2.plot(time_weeks, y_pd_best, color=PD_colors[i], label='PD simulation')

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()

            plt.suptitle('PK and PD Simulations in Skin of a SLE Patient', fontsize=22, fontweight='bold')
            ax1.set_title(f"{dose} mg/kg IV dose and pDC Density of {label} pDCs/mm²", fontsize=18)
            ax1.set_xlabel('Time [Weeks]', fontsize=18)
            ax1.set_ylabel('Free Litifilimab Skin Concentration (µg/mL)', color=PK_colors[i], fontsize=18)
            ax2.set_ylabel('Free BDCA2 Expression on pDCs (% Change from Baseline)', color=PD_colors[i], fontsize=18)
            ax1.tick_params(axis='y', labelcolor=PK_colors[i], which='major', labelsize=16)
            ax2.tick_params(axis='y', labelcolor=PD_colors[i], which='major', labelsize=16)
            ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=16)

            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            plt.tight_layout()

            save_plot(save_dir, f"IV_{dose}_PK_PD_sim_{label}")

    data_save_path = os.path.join(base_dir, 'Data', 'SLE_skin_IV_dose_PD_response_data.json')
    with open(data_save_path, "w") as f:
        json.dump(response_results, f, indent=4)


def plot_skin_PK_simulations(params, models, data, time_vectors):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PK')
    doses = {'IV_20_SLE': ('IV', 20), 'SC_50_SLE': ('SC', 50), 'SC_150_SLE': ('SC', 150), 'SC_450_SLE': ('SC', 450)}
    patient_labels = ['1', '10', '80', '400']
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, 4))

    for dose, (type, size) in doses.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        time_vector = time_vectors['IV'] if type == 'IV' else time_vectors['SC']
        time_weeks = time_vector / 168

        for i, label in enumerate(patient_labels):
            model = models[f'SLE_model_{label}']
            acceptable_params = params[f'acceptable_params_{label}']
            final_params = params[f'final_params'][f'SLE_{label}']

            sim = create_simulation_objects(model, 'SLE', bodyweight, dataset=data, custom_IV_doses=[size])

            y_min, y_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PK_skin_sim')
            y_best = simulate(final_params, sim, time_vector, 'PK_skin_sim')

            ax.fill_between(time_weeks, y_min, y_max, color=colors[i], alpha=0.3)
            ax.plot(time_weeks, y_best, color=colors[i], label=label)

        plt.suptitle('PK Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold', x=0.54)
        ax.set_xlabel('Time [Weeks]', fontsize = 18)
        ax.set_ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yscale('log')

        if dose == 'IV_20_SLE':
            ax.set_title(f"For a Single {size} mg/kg IV dose", fontsize = 18)
            ax.set_ylim(0.001, 100)
        else:
            ax.set_title(f"Repeated {size} mg SC doses", fontsize = 18)
            ax.set_ylim(0.0001, 100)

        plt.tight_layout()

        save_plot(save_dir, f"PK_skin_sim_{dose}")


def plot_skin_PD_simulations(params, models, data, time_vectors):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PD')
    doses = {'IVdose_20_SLE': ('IV', 20), 'SCdose_50_SLE': ('SC', 50), 'SCdose_150_SLE': ('SC', 150), 'SCdose_450_SLE': ('SC', 450)}
    patient_labels = ['1', '80', '400']
    colors = plt.cm.Reds(np.linspace(0.6, 0.9, 3))
    linestyles = ['--', '-', ':']

    for dose, (type, size) in doses.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        time_vector = time_vectors['IV'] if type == 'IV' else time_vectors['SC']
        time_weeks = time_vector / 168

        for i, label in enumerate(patient_labels):
            model = models[f'SLE_model_{label}']
            acceptable_params = params[f'acceptable_params_{label}']
            final_params = params[f'final_params'][f'SLE_{label}']
            
            if type == 'SC':
                sims = create_simulation_objects(model, 'SLE', bodyweight, dataset=data)
            else:
                sims = create_simulation_objects(model, 'SLE', bodyweight, custom_IV_doses=[size])

            y_min, y_max = calculate_uncertainty(sims[dose], time_vector, acceptable_params, 'SLE', 'PD_skin_sim')
            y_best = simulate(final_params, sims[dose], time_vector, 'PD_skin_sim')

            ax.fill_between(time_weeks, y_min, y_max, color=colors[i], alpha=0.5)
            ax.plot(time_weeks, y_best, color=colors[i], linestyle=linestyles[i], label=f"{label} pDCs/mm²")

        plt.suptitle('PD Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold', x=0.54)
        ax.set_xlabel('Time [Weeks]', fontsize = 18)
        ax.set_ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'lower right')
        
        if type == 'IV':
            ax.set_title(f"{dose} mg/kg IV dose", fontsize = 18)
        else:
            ax.set_title(f"Phase 2: {size} mg SC (W0, 2, 4 + Q4W to W20)", fontsize = 18)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        save_plot(save_dir, f"PD_skin_sim_{dose}")


def plot_IV_dose_response(data):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'Dose_response')
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Reds(np.linspace(0.6, 0.9, 3))
    linestyles = ['--', '-', ':']

    for i, (label, dataset) in enumerate(data.items()):
        color = colors[i]
        linestyle = linestyles[i]
        ax.fill_between(dataset['Dose'], dataset['Fast'], dataset['Slow'], color=color, alpha=0.5)
        ax.plot(dataset['Dose'], dataset['Best'], linestyle=linestyle, color=color, marker='o', linewidth=2, label=f"{label} pDCs/mm²")

    plt.suptitle("PD Response in Skin of SLE Patients", fontsize=22, fontweight='bold', x=0.52)
    ax.set_title("Time to 10 % Recovery from Peak BDCA2 Suppression", fontsize=18)
    ax.set_xlabel('IV Dose Size [mg/kg]', fontsize=18)
    ax.set_ylabel('Response Duration [Weeks]', fontsize=18)
    ax.legend(loc='upper left', fontsize=16, title='pDC Skin Density', title_fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 63)
    ax.set_ylim(0, 27)

    plt.tight_layout()

    save_plot(save_dir, "IV_dose_response")


def plot_skin_plasma_AUC_ratio(params, models, time_vector):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PK')

    patient_labels = ['1', '10', '80', '400']
    doses = [0.05, 0.3, 1, 3, 10, 20, 40, 60]
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, 4))

    AUC_results = {label: {} for label in patient_labels}

    for label in patient_labels:
        model = models[f'SLE_model_{label}']
        acceptable_params = params[f'acceptable_params_{label}']
        final_params = params[f'final_params'][f'SLE_{label}']
        for dose in doses:
            sim = create_simulation_objects(model, 'SLE', bodyweight, custom_IV_doses=[dose])

            plasma_min, plasma_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PK_sim')
            skin_min, skin_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PK_skin_sim')
            plasma_best = simulate(final_params, sim, time_vector, 'PK_sim')
            skin_best = simulate(final_params, sim, time_vector, 'PK_skin_sim')

            AUC_results[label][dose] = {
                "best": 100 * np.trapezoid(skin_best, time_vector) / (np.trapezoid(plasma_best, time_vector) if np.trapezoid(plasma_best, time_vector) != 0 else np.nan),
                "min": 100 * np.trapezoid(skin_min, time_vector) / (np.trapezoid(plasma_max, time_vector) if np.trapezoid(plasma_max, time_vector) != 0 else np.nan),
                "max": 100 * np.trapezoid(skin_max, time_vector) / (np.trapezoid(plasma_min, time_vector) if np.trapezoid(plasma_min, time_vector) != 0 else np.nan)
            } 

    txt_path = os.path.join(save_dir, "AUC_skin_plasma_ratios.txt")
    with open(txt_path, "w") as f:
        header = f"{'Patient':<10} {'Dose':<6} {'AUC_best_ratio (%)':>20} {'AUC_min_ratio (%)':>20} {'AUC_max_ratio (%)':>20}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for label in patient_labels:
            for dose in doses:
                res = AUC_results[label][dose]
                f.write(f"{label:<10} {dose:<6} {res['best']:20.3f} {res['min']:20.3f} {res['max']:20.3f}\n")  
    
    fig, ax = plt.subplots(figsize=(10,8))
    bar_width = 0.2
    x = np.arange(len(doses))

    for i, label in enumerate(patient_labels):
        best = [AUC_results[label][dose]["best"] for dose in doses]
        min = [AUC_results[label][dose]["min"] for dose in doses]
        max = [AUC_results[label][dose]["max"] for dose in doses]

        yerr = [np.array(best) - np.array(min), np.array(max) - np.array(best)]
        ax.bar(x + i * bar_width, best, width=bar_width, color=colors[i], label=f"{label} pDCs/mm²", yerr=yerr, capsize=6)

    plt.suptitle('AUC Ratio of Litifilimab in Skin vs Plasma', fontsize=22, fontweight='bold', x=0.52)
    ax.set_title('Simulated over 24 Weeks in SLE Patients', fontsize=18)
    ax.set_xlabel('IV Dose Size [mg/kg]', fontsize=18)
    ax.set_ylabel('AUC Ratio Skin vs Plasma [%]', fontsize=18)
    ax.legend(title='pDC Skin Density', title_fontsize=18, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks(x + (bar_width * (len(patient_labels)-1) / 2))
    ax.set_xticklabels([f"{dose}" for dose in doses], fontsize=16) 

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    ax.set_ylim(0.001, 100)

    plt.tight_layout()
    save_plot(save_dir, "AUC_skin_plasma_ratios")


def plot_skin_plasma_concentration_ratio(params, models, time_vector):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PK')
    os.makedirs(save_dir, exist_ok=True)

    # Scenarios we want to compare
    scenarios = {
        "SLE Patient (10 pDCs/mm² in Skin)": ('SLE_model_10', '10'),
        "SLE Patient (80 pDCs/mm² in Skin)": ('SLE_model_80', '80'),
        "HV (12000 pDCs/mL in Blood)":       ('HV_model_high', '80'), # Using 80 as proxy for HV acc_params
        "HV (5100 pDCs/mL in Blood)":        ('HV_model', '80')
    }
    
    blue_shades = plt.cm.Blues(np.linspace(0.7, 0.9, 2))
    common_plasma_range = np.logspace(-3, 3, 500)
    start_index = np.searchsorted(time_vector, 12)

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (label, (m_key, p_suffix)) in enumerate(scenarios.items()):
        model = models[m_key]
        p_type = 'HV' if 'HV' in m_key else 'SLE'
        
        # Access the main params dict internally
        best_p = params['final_params']['HV' if p_type == 'HV' else f'SLE_{p_suffix}']
        acc_p = params[f'acceptable_params_{p_suffix}']

        sim_dict = create_simulation_objects(model, p_type, bodyweight, custom_IV_doses=[20])
        sim = list(sim_dict.values())[0]

        # Uncertainty Calculation
        all_interp_skin = []
        for p in acc_p:
            adj_p = np.delete(np.array(p).copy(), [11, 16] if p_type == 'HV' else [10, 15])
            try:
                plasma = simulate(adj_p, sim, time_vector, 'PK_sim')[start_index:]
                skin = simulate(adj_p, sim, time_vector, 'PK_skin_sim')[start_index:]
                sort_idx = np.argsort(plasma)
                all_interp_skin.append(np.interp(common_plasma_range, plasma[sort_idx], skin[sort_idx], left=np.nan, right=np.nan))
            except RuntimeError: continue

        if all_interp_skin:
            ax.fill_between(common_plasma_range, np.nanmin(all_interp_skin, axis=0), np.nanmax(all_interp_skin, axis=0), 
                             color=blue_shades[0] if p_type == 'HV' else blue_shades[1], alpha=0.3)

        # Best Fit Calculation
        adj_best = np.delete(np.array(best_p).copy(), [11, 16] if p_type == 'HV' else [10, 15])
        plasma_best = simulate(adj_best, sim, time_vector, 'PK_sim')[start_index:]
        skin_best = simulate(adj_best, sim, time_vector, 'PK_skin_sim')[start_index:]
        ax.plot(plasma_best, skin_best, label=label, color=blue_shades[0] if p_type == 'HV' else blue_shades[1], 
                linestyle='--' if i % 2 != 0 else '-', linewidth=3)

    # Literature references
    ref_x = np.logspace(-3, 3, 100) 
    ax.plot(ref_x, 0.157 * ref_x, 'k-', linewidth=3, label='Skin Distribution in Literature (15.7 %)')
    ax.plot(ref_x, 0.0785 * ref_x, 'k--', linewidth=2, label='2-fold Error')
    ax.plot(ref_x, 0.314 * ref_x, 'k--', linewidth=2)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-3, 4e2); ax.set_ylim(1e-4, 1e2)
    ax.set_xlabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
    ax.set_ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize=18)
    ax.set_title(f'For a 20 mg/kg IV Dose in HV and SLE Patients', fontsize=18)
    plt.suptitle('Distribution of Litifilimab in Skin vs Plasma', fontsize=22, fontweight='bold', x=0.54)
    ax.legend(fontsize=16, loc='upper left'); ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_plot(save_dir, "Skin_vs_plasma_distribution")


def simulate_SC_dose_response_frequency(params, model, density_label):
    doses_mg = [50, 150, 300, 450, 600]
    treatment_duration_w = 25 
    threshold = -90 
    
    # Internal parameter extraction
    best_p = np.delete(np.array(params['final_params'][f'SLE_{density_label}']).copy(), [10, 15])
    acc_p = params[f'acceptable_params_{density_label}']
    
    results = {"Dose": doses_mg, "Best": [], "Fast": [], "Slow": []}
    
    for dose_mg in doses_mg:
        for search_type in ['Best', 'Fast', 'Slow']:
            low, high = 1, int(treatment_duration_w * 168)
            found_h = 0
            while low <= high:
                mid_h = (low + high) // 2
                sim = list(create_simulation_objects(model, 'SLE', bodyweight, 
                           custom_SC_doses=[{'size_mg': dose_mg, 'interval_weeks': mid_h/168.0, 'total_weeks': treatment_duration_w}]).values())[0]
                
                t_vec = np.arange(0, treatment_duration_w * 168 + 1, 1)
                if search_type == 'Best':
                    passed = check_suppression_maintained(simulate(best_p, sim, t_vec, 'PD_skin_sim'), t_vec, threshold, treatment_duration_w * 168)
                else:
                    y_min, y_max = calculate_uncertainty(sim, t_vec, acc_p, 'SLE', 'PD_skin_sim')
                    passed = check_suppression_maintained(y_max if search_type == 'Fast' else y_min, t_vec, threshold, treatment_duration_w * 168)
                
                if passed: found_h = mid_h; low = mid_h + 1
                else: high = mid_h - 1
            results[search_type].append(round(found_h / 168.0, 3))

    with open(os.path.join(base_dir, 'Data', f"SLE_skin_SC_dose_response_{density_label}.json"), "w") as f:
        json.dump(results, f, indent=4)


def plot_SC_dose_response(SC_dose_response_data, inverse=False):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'Dose_response')
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Reds(np.linspace(0.6, 0.9, len(SC_dose_response_data)))
    linestyles = ['--', '-', ':']

    for i, (label, dataset) in enumerate(SC_dose_response_data.items()):
        color = colors[i]
        linestyle = linestyles[i % len(linestyles)]
        
        # Convert intervals to frequencies
        y_best = 1.0 / np.array(dataset['Best']) if inverse else np.array(dataset['Best'])
        y_fast = 1.0 / np.array(dataset['Fast']) if inverse else np.array(dataset['Fast'])
        y_slow = 1.0 / np.array(dataset['Slow']) if inverse else np.array(dataset['Slow'])
        
        plt.fill_between(dataset['Dose'], y_slow, y_fast, color=color, alpha=0.5)
        plt.plot(dataset['Dose'], y_best, linestyle=linestyle, color=color, linewidth=2, marker='o', label=f"{label} pDCs/mm²")

    # Axis Formatting
    ax = plt.gca()
    plt.xlabel('SC Dose Size [mg]', fontsize=18)
    plt.ylabel('Dosing Frequency [Doses/Week]' if inverse else 'Max Interval Between Doses [Weeks]', fontsize=18)
    
    if inverse:
        plt.yscale('log')
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')
    
    plt.suptitle("Required SC Dosing Frequency for SLE Patients" if inverse else "Required SC Dosing Interval for SLE Patients", fontsize=22, fontweight='bold', x=0.54)
    ax.set_title(r"To Maintain $\geq$90% BDCA2 Suppression in Skin" if inverse else r"To Maintain $\geq$90% BDCA2 Suppression in Skin", fontsize=18)
    plt.legend(loc='upper right' if inverse else 'upper left', fontsize=16, title='pDC Skin Density', title_fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_plot(save_dir, "SC_dose_response_inverse" if inverse else "SC_dose_response")


def plot_PD_SC_frequency(params, models, SC_dose_response_data):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PD', 'SC_Frequency')
    doses_mg = [50, 150, 300, 450, 600]
    scenarios = ['Fast', 'Best', 'Slow']
    colors = plt.cm.Reds(np.linspace(0.7, 0.9, 3)) 
    linestyles = [':', '-', '--']
    
    for i, (density, intervals) in enumerate(SC_dose_response_data.items()):
        model = models[f'SLE_model_{density}']
        best_p = params['final_params'][f'SLE_{density}']
        acc_p = params[f'acceptable_params_{density}']

        for d_idx, dose_mg in enumerate(doses_mg):
            # fig, ax = plt.subplots(figsize=(10, 8))
            for s_idx, scenario in enumerate(scenarios):
                fig, ax = plt.subplots(figsize=(10, 8))
                interval = intervals[scenario][d_idx]
                if interval <= 0: continue
                
                sim = list(create_simulation_objects(model, 'SLE', bodyweight, 
                           custom_SC_doses=[{'size_mg': dose_mg, 'interval_weeks': interval, 'total_weeks': 25}]).values())[0]
                
                t_vec = np.arange(0, 7500, 1)
                y_min, y_max = calculate_uncertainty(sim, t_vec, acc_p, 'SLE', 'PD_skin_sim')
                y_best = simulate(best_p, sim, t_vec, 'PD_skin_sim')

                rounded_interval = round(7 * interval / 0.5) * 0.5

                suffix = "~Daily" if rounded_interval == 1 else f"Every ~{rounded_interval} Days"
                
                ax.fill_between(t_vec/168, y_min, y_max, color=colors[i], alpha=0.5)
                ax.plot(t_vec/168, y_best, color=colors[i], linestyle=linestyles[i], label=f"{density} pDCs/mm²")

                plt.axvline(x=10, color='k', linestyle=':', alpha = 0.8, label='10 Week Threshold') 
                ax.axhline(y=-90, color='k', linestyle='--', alpha=0.8, label='90 % Suppression Threshold')
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_xlabel('Time [Weeks]', fontsize=18)
                ax.set_ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize = 18)
                plt.suptitle('Personalized Dosing Protocol for Sustained Response', fontsize = 22, fontweight='bold', x=0.50)
                ax.set_title(f"{dose_mg} mg SC: {round(1.0/interval, 2)} Doses/Week ({suffix})", fontsize=18)
                ax.legend(title='PD Simulation in Skin', fontsize=16, loc='upper left', title_fontsize=18)
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            
                plt.tight_layout()
                save_plot(save_dir, f"PD_skin_sim_SC_frequency_{dose_mg}mg_{density}_{scenario}")


# plot_skin_PK_PD_simulations_together(params, models, time_vectors['IV'])

# plot_skin_PK_simulations(params, models, data['SLE_CLE_PK_validation_data'], time_vectors)

# plot_skin_PD_simulations(params, models, data['SLE_CLE_PK_validation_data'], time_vectors)

# plot_IV_dose_response(data['SLE_IV_dose_skin_PD_response_data'])

# plot_skin_plasma_AUC_ratio(params, models, time_vectors['AUC'])

# plot_skin_plasma_concentration_ratio(params, models, time_vectors['Ratio'])

# for density in ['1', '10', '80', '400']:
#     simulate_SC_dose_response_frequency(params, models[f'SLE_model_{density}'], density)

# plot_SC_dose_response(data['SLE_SC_dose_skin_PD_response_data'], inverse=False)

# plot_SC_dose_response(data['SLE_SC_dose_skin_PD_response_data'], inverse=True)

plot_PD_SC_frequency(params, models, data['SLE_SC_dose_skin_PD_response_data'])


# def plot_skin_plasma_concentration_ratio(best_param_sets, acceptable_param_sets, models, time_vector, save_dir='../../../Results/SLE/PK'):
#     os.makedirs(save_dir, exist_ok=True)

#     doses = ['IV_20_SLE', 'IV_20_SLE', 'IV_20_SLE', 'IV_20_SLE']
#     labels = ["SLE Patient (10 pDCs/mm² in Skin)", "SLE Patient (80 pDCs/mm² in Skin)", "HV (12000 pDCs/mL in Blood)", "HV (5100 pDCs/mL in Blood)"]
#     blue_shades= plt.cm.Blues(np.linspace(0.7, 0.9, 2))
#     HV_color = blue_shades[0]
#     SLE_color = blue_shades[1]
#     colors = [SLE_color, SLE_color, HV_color, HV_color]
#     linestyles = ['-', '--', '-', '--']
    

#     fig, ax = plt.subplots(figsize=(10, 8))

#     common_plasma_range = np.logspace(-3, 3, 500)

#     timepoints = time_vector
#     start_index = np.searchsorted(timepoints, 12)

#     for (patient_label, best_params), model, acceptable_params, dose, color, label, linestyle in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), doses, colors, labels, linestyles):
#         sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h')}

#         all_interp_skin = []

#         for acceptable_param in acceptable_params:
#             if patient_label in ['HV', 'HV_high']:
#                 params = np.delete(acceptable_param.copy(), [11,16])
#             else:
#                 params = np.delete(acceptable_param.copy(), [10,15])

#             try:
#                 sims[dose].simulate(time_vector=timepoints, parameter_values=params, reset=True)
#                 plasma = sims[dose].feature_data[start_index:, 0]
#                 skin = sims[dose].feature_data[start_index:, 2]

#                 sort_idx = np.argsort(plasma)
#                 interp_skin = np.interp(common_plasma_range, plasma[sort_idx], skin[sort_idx], left=np.nan, right=np.nan)
#                 all_interp_skin.append(interp_skin)
#             except RuntimeError:
#                 continue

#         if all_interp_skin:
#             y_min = np.nanmin(all_interp_skin, axis=0)
#             y_max = np.nanmax(all_interp_skin, axis=0)
#             plt.fill_between(common_plasma_range, y_min, y_max, color=color, alpha=0.3)

#         if patient_label in ['HV', 'HV_high']:
#             params_best = np.delete(best_params.copy(), [11,16])
#         else:   
#             params_best = np.delete(best_params.copy(), [10,15])
            
#         sims[dose].simulate(time_vector=timepoints, parameter_values=params_best, reset=True)
#         x_best_plasma = sims[dose].feature_data[start_index:, 0]
#         y_best_skin = sims[dose].feature_data[start_index:, 2]

#         if patient_label in ['HV', 'HV_high']:
#             plt.plot(x_best_plasma, y_best_skin, color=color, label=f"{label}", linestyle=linestyle, linewidth=3)
#         else:
#             plt.plot(x_best_plasma, y_best_skin, color=color, label=f"{label}", linestyle=linestyle, linewidth=3)

#     ref_x = np.logspace(-3, 3, 100) 
#     plt.plot(ref_x, 0.157 * ref_x, 'k-', linewidth=3, label='Skin Distribution in Literature (15.7 %)')
#     plt.plot(ref_x, 0.0785 * ref_x, 'k--', linewidth=2, label='2-fold Error')
#     plt.plot(ref_x, 0.314 * ref_x, 'k--', linewidth=2)

#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlim(1e-3, 4e2) 
#     plt.ylim(1e-4, 1e2)
#     plt.xlabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
#     plt.ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize=18)
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.title(f'For a 20 mg/kg IV Dose in HV and SLE Patients', fontsize=18)
#     plt.suptitle('Distribution of Litifilimab in Skin vs Plasma', fontsize=22, fontweight='bold', x=0.54)
#     plt.legend(fontsize=16, loc='upper left')
#     plt.tick_params(axis='both', which='major', labelsize=16)
#     plt.tight_layout()

#     # Save the plot
#     save_path_png = os.path.join(save_dir, f"Skin_vs_plasma_distribution.png")
#     plt.savefig(save_path_png, format='png', dpi=600)
#     save_path_svg = os.path.join(save_dir, "Skin_vs_plasma_distribution.svg")
#     plt.savefig(save_path_svg, format='svg')
#     plt.close()



# def simulate_SC_dose_response_frequency(density_label, best_param_sets, acceptable_param_sets, models):
#     save_dir = '../../../Results/SLE/SC_Frequency'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Configuration
#     doses_mg = [50, 150, 300, 450, 600]
#     treatment_duration_weeks = 25 
#     threshold = -90 
#     hours_per_week = 168.0
    
#     # Define the search bounds in hours
#     # From 1 hour up to the full treatment duration (e.g., 60 weeks)
#     MAX_HOURS = int(treatment_duration_weeks * hours_per_week)

#     model = models[density_label]
#     acceptable_params = acceptable_param_sets[density_label]
#     SLE_best_params = np.delete(best_param_sets[density_label].copy(), [10, 15])
    
#     results = {"Dose": doses_mg, "Best": [], "Fast": [], "Slow": []}
    
#     for dose_mg in doses_mg:
#         print(f"--- Searching Optimal Frequency for {dose_mg} mg (Density: {density_label}) ---")

#         def evaluate_h_interval(h):
#             """Returns (best_pass, fast_pass, slow_pass) for a given hour interval."""
#             interval_w = h / hours_per_week
            
#             # SAFETY: If the interval is shorter than the infusion duration, it's invalid
#             if h <= 0.25: 
#                 return False, False, False
                
#             activity, m_end = create_SC_activity(dose_mg, interval_w, treatment_duration_weeks)
#             t_vec = np.arange(0, m_end + 1, 1)
#             sim = sund.Simulation(models=model, activities=activity, time_unit='h')
            
#             # Initialize pass flags
#             pass_best, pass_fast, pass_slow = False, False, False
            
#             # 1. Simulate Best with Error Handling
#             try:
#                 sim.simulate(time_vector=t_vec, parameter_values=SLE_best_params, reset=True)
#                 pass_best = check_suppression_maintained(sim.feature_data[:, 3], t_vec, threshold, m_end)
#             except RuntimeError:
#                 pass_best = False # Treat solver failure as a failure to maintain suppression

#             # 2. Simulate Uncertainty with Error Handling
#             y_min = np.full_like(t_vec, 10000)
#             y_max = np.full_like(t_vec, -10000)
#             sim_success_count = 0

#             for acc_p in acceptable_params:
#                 p = np.delete(acc_p.copy(), [10, 15])
#                 try:
#                     sim.simulate(time_vector=t_vec, parameter_values=p, reset=True)
#                     y_sim = sim.feature_data[:, 3]
#                     y_min = np.minimum(y_min, y_sim)
#                     y_max = np.maximum(y_max, y_sim)
#                     sim_success_count += 1
#                 except RuntimeError:
#                     continue
            
#             # Only evaluate uncertainty if at least some simulations succeeded
#             if sim_success_count > 0:
#                 pass_fast = check_suppression_maintained(y_max, t_vec, threshold, m_end)
#                 pass_slow = check_suppression_maintained(y_min, t_vec, threshold, m_end)
            
#             return pass_best, pass_fast, pass_slow

#         # Binary Search for 'Best'
#         low, high = 1, MAX_HOURS
#         best_h = 0
#         while low <= high:
#             mid = (low + high) // 2
#             p_best, _, _ = evaluate_h_interval(mid)
#             if p_best:
#                 best_h = mid
#                 low = mid + 1
#             else:
#                 high = mid - 1
        
#         # Binary Search for 'Fast' (Least Suppressed)
#         low, high = 1, MAX_HOURS
#         fast_h = 0
#         while low <= high:
#             mid = (low + high) // 2
#             _, p_fast, _ = evaluate_h_interval(mid)
#             if p_fast:
#                 fast_h = mid
#                 low = mid + 1
#             else:
#                 high = mid - 1

#         # Binary Search for 'Slow' (Most Suppressed)
#         low, high = 1, MAX_HOURS
#         slow_h = 0
#         while low <= high:
#             mid = (low + high) // 2
#             _, _, p_slow = evaluate_h_interval(mid)
#             if p_slow:
#                 slow_h = mid
#                 low = mid + 1
#             else:
#                 high = mid - 1

#         results["Best"].append(round(best_h / hours_per_week, 3))
#         results["Fast"].append(round(fast_h / hours_per_week, 3))
#         results["Slow"].append(round(slow_h / hours_per_week, 3))
#         print(f"Result for {dose_mg}mg: Best={results['Best'][-1]}w, Fast={results['Fast'][-1]}w, Slow={results['Slow'][-1]}w")

#     # Save unique file for this density
#     result_filename = f"../../../Data/SLE_skin_SC_dose_response_{density_label}.json"
#     with open(result_filename, "w") as f:
#         json.dump(results, f, indent=4)
#     print(f"FINISHED: Results saved to {result_filename}")




# def plot_SC_dose_response(SC_dose_response_data):
#     """
#     Plots gathered SC data: Dose (X) vs. Max Allowed Interval (Y).
#     Assumes all_sc_results is a dictionary containing data for 
#     densities 1, 10, 80, and 400.
#     """
#     save_dir = '../../../Results/SLE/Dose_response'
#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure(figsize=(10, 8))

#     # Using the Red colormap for consistency with PD themes
#     colors = plt.cm.Purples(np.linspace(0.6, 0.9, len(SC_dose_response_data)))
#     # Standard 4 linestyles plus a 5th custom style for density 400 if needed
#     linestyles = ['--', '-', ':', '-.', (0, (3, 5, 1, 5))]

#     for i, (label, dataset) in enumerate(SC_dose_response_data.items()):
#         color = colors[i]
#         linestyle = linestyles[i % len(linestyles)]
        
#         # Plot the uncertainty range (Fast vs Slow responders)
#         plt.fill_between(dataset['Dose'], dataset['Fast'], dataset['Slow'], 
#                          color=color, alpha=0.3)
        
#         # Plot the Best parameter line
#         plt.plot(dataset['Dose'], dataset['Best'], linestyle=linestyle, color=color, 
#                  linewidth=2, marker='o', label=f"{label} pDCs/mm²")

#     plt.xlabel('SC Dose Size [mg]', fontsize=18)
#     plt.ylabel('Maximum Interval Between Doses [Weeks]', fontsize=18)
#     plt.title('Continuous Suppression of >90% BDCA2 on pDCs in Skin', fontsize=18)
#     plt.suptitle("Required Frequency of SC Doses to Sustain Response", fontsize=22, fontweight='bold', x=0.52)
#     plt.tick_params(axis='both', which='major', labelsize=16)
#     plt.legend(loc='upper left', fontsize=16, title='pDC Skin Density', title_fontsize=18)
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
#     # Cap Y-axis to your max search range (e.g., 60 weeks)
#     plt.ylim(0, 16) 
#     plt.tight_layout()

#     plt.savefig(os.path.join(save_dir, "SC_dose_response.png"), dpi=600)
#     plt.savefig(os.path.join(save_dir, "SC_dose_response.svg"), format='svg')
#     plt.close()



# def plot_SC_dose_response_inverse(SC_dose_response_data):
#     """
#     Plots gathered SC data with Dose on X and Frequency (1/Interval) on Y.
#     Higher Y values = Higher frequency (more doses per week).
#     """
#     save_dir = '../../../Results/SLE/Dose_response'
#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure(figsize=(10, 8))

#     colors = plt.cm.Purples(np.linspace(0.6, 0.9, len(SC_dose_response_data)))
#     linestyles = ['--', '-', ':', '-.']

#     for i, (label, dataset) in enumerate(SC_dose_response_data.items()):
#         color = colors[i]
#         linestyle = linestyles[i % len(linestyles)]
        
#         # Convert intervals (weeks) to frequencies (1/weeks)
#         # Using numpy to handle element-wise division safely
#         best_inv = 1.0 / np.array(dataset['Best'])
#         fast_inv = 1.0 / np.array(dataset['Fast'])
#         slow_inv = 1.0 / np.array(dataset['Slow'])
        
#         # Replace infinity (from division by zero) with 0 for plotting
#         best_inv[np.isinf(best_inv)] = 0
#         fast_inv[np.isinf(fast_inv)] = 0
#         slow_inv[np.isinf(slow_inv)] = 0

#         # Plot uncertainty: slow_inv is the lower freq, fast_inv is the higher freq
#         plt.fill_between(dataset['Dose'], slow_inv, fast_inv, 
#                          color=color, alpha=0.3)
        
#         # Plot the Best frequency
#         plt.plot(dataset['Dose'], best_inv, linestyle=linestyle, color=color, 
#                  linewidth=2, marker='o', label=f"{label} pDCs/mm²")

#     plt.xlabel('SC Dose Size [mg]', fontsize=18)
#     plt.ylabel('Minimum Dosing-Frequency [Doses/Week]', fontsize=18)
#     plt.title('Dosing to Sustain >90% BDCA2 Suppression in Skin', fontsize=18)
#     plt.suptitle("Required SC Dosing Frequency", fontsize=22, fontweight='bold', x=0.52)
#     plt.tick_params(axis='both', which='major', labelsize=16)
#     plt.legend(loc='upper right', fontsize=16, title='pDC Skin Density', title_fontsize=18)
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
    
#     # Optional: Use log scale if frequencies span multiple orders of magnitude
#     plt.yscale('log') 
    
#     plt.tight_layout()

#     plt.savefig(os.path.join(save_dir, "SC_dose_response_inverse.png"), dpi=600)
#     plt.savefig(os.path.join(save_dir, "SC_dose_response_inverse.svg"), format='svg')
#     plt.close()


# def plot_PD_SC_frequency(best_param_sets, acceptable_param_sets, models, SC_dose_response_data):
#     save_dir = '../../../Results/SLE/PD/SC_Frequency'
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Configuration
#     doses_mg = [50, 150, 300, 450, 600]
#     treatment_duration_weeks = 25 
#     threshold = -90
#     hours_per_week = 168.0
    
#     # Distinct colors and labels for our three critical scenarios
#     scenarios = ['Fast', 'Best', 'Slow']
#     colors = plt.cm.Reds(np.linspace(0.7, 0.9, 3)) 
#     linestyles = [':', '-', '--']
    
#     for density_label, best_params in best_param_sets.items():
        
#         model = models[density_label]
#         acceptable_params = acceptable_param_sets[density_label]
#         SLE_best_params = np.delete(best_params.copy(), [10, 15])
        
#         # Get the results found during the binary search for this density
#         found_intervals = SC_dose_response_data[density_label]
        
#         for dose_idx, dose_mg in enumerate(doses_mg):
#             plt.figure(figsize=(10, 8))
            
#             for s_idx, scenario in enumerate(scenarios):
#                 interval = found_intervals[scenario][dose_idx]
                
#                 if interval <= 0:
#                     continue
                
#                 # INVERSION: Calculate doses per week
#                 frequency = round(1.0 / interval, 2)
                
#                 color = colors[s_idx]
#                 linestyle = linestyles[s_idx]
                
#                 # Create activity for this specific scenario interval
#                 activity, maintenance_end = create_SC_activity(dose_mg, interval, treatment_duration_weeks)
                
#                 time_vector = np.arange(0, maintenance_end + 2200, 1)
#                 time_weeks = time_vector / hours_per_week
                
#                 sim = sund.Simulation(models=model, activities=activity, time_unit='h')
                
#                 y_min = np.full_like(time_vector, 10000)
#                 y_max = np.full_like(time_vector, -10000)

#                 # Simulate Uncertainty
#                 for acceptable_param in acceptable_params:
#                     SLE_params = np.delete(acceptable_param.copy(), [10, 15])
#                     try:
#                         sim.simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
#                         y_sim = sim.feature_data[:, 3]
#                         y_min = np.minimum(y_min, y_sim)
#                         y_max = np.maximum(y_max, y_sim)
#                     except RuntimeError:
#                         continue

#                 # Simulate and Plot the Best fit
#                 sim.simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
#                 y_best = sim.feature_data[:, 3]
                
#                 # Updated Label to "x doses per week"
#                 label_text = f"{frequency} Doses/Week"
                
#                 plt.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.2)
#                 plt.plot(time_weeks, y_best, color=color, linestyle=linestyle, 
#                          linewidth=2, label=label_text)

#             # Formatting
#             plt.axhline(y=threshold, color='k', linestyle='--',alpha = 0.8, label='90% Threshold')
#             plt.axvline(x=10, color='k', linestyle=':', alpha = 0.8, label='10-Week Threshold') 
            
#             plt.xlabel('Time [Weeks]', fontsize=18)
#             plt.ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize=18)
#             plt.title(f"For Repeated {dose_mg} mg SC Doses and {density_label} pDCs/mm² in Skin", fontsize=18)
#             plt.suptitle(f'PD Simulations in Skin of SLE Patient', fontsize=22, fontweight='bold', x=0.54)
            
#             plt.gca().spines['top'].set_visible(False)
#             plt.gca().spines['right'].set_visible(False)
#             plt.tick_params(axis='both', which='major', labelsize=16)
#             plt.legend(title='Frequency of SC Doses', title_fontsize=18, fontsize=16, loc='upper left')
#             plt.tight_layout()

#             plt.savefig(os.path.join(save_dir, f"PD_skin_sim_SC_frequency_{dose_mg}mg_{density_label}.png"), dpi=600)
#             plt.savefig(os.path.join(save_dir, f"PD_skin_sim_SC_frequency_{dose_mg}mg_{density_label}.svg"))
#             plt.close()

