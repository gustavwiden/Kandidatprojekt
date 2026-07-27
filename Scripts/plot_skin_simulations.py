import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import json

from utils import base_dir, load_data, load_models, load_params, create_simulation_objects, calculate_uncertainty, simulate, save_plot, get_response_time, check_suppression_maintained

data = load_data('SLE_CLE_PK_validation_data', 'SLE_IV_dose_skin_PD_response_data', 'SLE_SC_dose_skin_PD_response_data')  

models = load_models('HV_model', 'SLE_model_1', 'SLE_model_80', 'SLE_model_400', 'HV_model_high')    

params = load_params('final_params', 'acceptable_params_1', 'acceptable_params_80', 'acceptable_params_400')

# Overall average bodyweight for healthy volunteers (cohort 1-7) and SLE patients (cohort 8) in the phase 1 trial
bodyweight = 73

time_vectors = {'IV': np.arange(-10, 7000, 1), 'SC': np.arange(-10, 7500, 1), 'AUC': np.arange(0, 2688, 1), 'Ratio': np.arange(0, 10000, 1)}

density_labels = ['1','80','400']

# Generate plot for Figure 5C
def plot_IV_dose_response(params, models, time_vector):
    patient_labels = ['1', '80', '400']
    doses = np.arange(0.05, 61, 1)
    data_save_path = os.path.join(base_dir, 'Data', 'SLE_skin_IV_dose_PD_response_data.json')
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'Dose_response')

    response_results = {label: {"Dose": doses.tolist(), "Best": [], "Fast": [], "Slow": []} for label in patient_labels}
    time_weeks = time_vector / 168

    for i, label in enumerate(patient_labels):
        model = models[f'SLE_model_{label}']
        acceptable_params = params[f'acceptable_params_{label}']
        final_params = params[f'final_params'][f'SLE_{label}']

        for dose in doses:
            sim = create_simulation_objects(model, 'SLE', bodyweight, custom_IV_doses=[dose])

            y_pd_min, y_pd_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PD_skin_sim')
            y_pd_best = simulate(final_params, sim, time_vector, 'PD_skin_sim')

            response_threshold = -90
            startpoint = 0

            response_results[label]["Best"].append(get_response_time(y_pd_best, response_threshold, startpoint, time_weeks, 'PD'))
            response_results[label]["Fast"].append(get_response_time(y_pd_max, response_threshold, startpoint, time_weeks, 'PD'))
            response_results[label]["Slow"].append(get_response_time(y_pd_min, response_threshold, startpoint, time_weeks, 'PD'))

    with open(data_save_path, "w") as f:
        json.dump(response_results, f, indent=4)
    
    with open(data_save_path, "r") as f:
        plot_data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Reds(np.linspace(0.6, 0.9, 3))
    linestyles = ['--', '-', ':']

    for i, (label, dataset) in enumerate(plot_data.items()):
        color = colors[i]
        linestyle = linestyles[i]
        ax.fill_between(dataset['Dose'], dataset['Fast'], dataset['Slow'], color=color, alpha=0.5, label=label)
        ax.plot(dataset['Dose'], dataset['Best'], linestyle=linestyle, color=color, linewidth=2, label=label)

    plt.suptitle("PD Response in Skin of SLE Patients", fontsize=22, fontweight='bold', x=0.52)
    ax.set_title("Time to 10% Recovery from Maximal BDCA2 Suppression", fontsize=18)
    ax.set_xlabel('IV Dose Size [mg/kg]', fontsize=18)
    ax.set_ylabel('Response Duration [Weeks]', fontsize=18)
    ax.legend(loc='upper left', fontsize=16, ncols=3)
    ax.tick_params(axis='both', which='major', labelsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 62)
    ax.set_ylim(0, 27)

    plt.tight_layout()

    save_plot(save_dir, "IV_dose_response")

# Generate plots for Figure 5A and Supplementary Figure 3A-C
def plot_skin_PK_simulations(params, models, data, time_vectors):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PK')
    doses = {'IVdose_20_SLE': ('IV', 20), 'SCdose_50_SLE': ('SC', 50), 'SCdose_150_SLE': ('SC', 150), 'SCdose_450_SLE': ('SC', 450)}
    patient_labels = ['1', '80', '400']
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, 3))
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

            y_min, y_max = calculate_uncertainty(sims[dose], time_vector, acceptable_params, 'SLE', 'PK_skin_sim')
            y_best = simulate(final_params, sims[dose], time_vector, 'PK_skin_sim')

            ax.fill_between(time_weeks, y_min, y_max, color=colors[i], alpha=0.5)
            ax.plot(time_weeks, y_best, color=colors[i], linestyle=linestyles[i], label=f"{label} pDCs/mm²")
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.8)

        plt.suptitle('PK Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold', x=0.54)
        ax.set_xlabel('Time [Weeks]', fontsize = 18)
        ax.set_ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yscale('log')

        if dose == 'IVdose_20_SLE':
            ax.set_title(f"{size} mg/kg IV dose", fontsize = 18)
            ax.set_ylim(0.001, 100)
        else:
            ax.set_title(f"Phase 2: {size} mg SC (W0, 2, 4 + Q4W to W20)", fontsize = 18)
            ax.set_ylim(0.0001, 100)

        plt.tight_layout()

        save_plot(save_dir, f"PK_skin_sim_{dose}")

# Generate plots for Figure 5B and Supplementary Figure 3D-F
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
        ax.axhline(y=-90, color='k', linestyle='--', alpha=0.8)

        plt.suptitle('PD Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold', x=0.54)
        ax.set_xlabel('Time [Weeks]', fontsize = 18)
        ax.set_ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize = 18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'upper right', bbox_to_anchor=(0.97, 0.9))
        
        if type == 'IV':
            ax.set_title(f"{size} mg/kg IV dose", fontsize = 18)
        else:
            ax.set_title(f"Phase 2: {size} mg SC (W0, 2, 4 + Q4W to W20)", fontsize = 18)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        save_plot(save_dir, f"PD_skin_sim_{dose}")

# Generate plot for Figure 7A
def plot_skin_plasma_AUC_ratio(params, models, time_vector):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PK')

    patient_labels = ['1', '80', '400']
    doses = [0.05, 0.3, 1, 3, 10, 20, 40, 60]
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, 3))

    AUC_results = {label: {} for label in patient_labels}

    for label in patient_labels:
        model = models[f'SLE_model_{label}']
        acceptable_params = params[f'acceptable_params_{label}']
        final_params = params[f'final_params'][f'SLE_{label}']
        for dose in doses:
            sim = create_simulation_objects(model, 'SLE', bodyweight, custom_IV_doses=[dose])

            plasma_min, plasma_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PK_plasma_sim')
            skin_min, skin_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PK_skin_sim')
            plasma_best = simulate(final_params, sim, time_vector, 'PK_plasma_sim')
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
    bar_width = 0.25
    x = np.arange(len(doses))

    for i, label in enumerate(patient_labels):
        best = [AUC_results[label][dose]["best"] for dose in doses]
        min = [AUC_results[label][dose]["min"] for dose in doses]
        max = [AUC_results[label][dose]["max"] for dose in doses]

        yerr = [np.array(best) - np.array(min), np.array(max) - np.array(best)]
        ax.bar(x + i * bar_width, best, width=bar_width, color=colors[i], label=f"{label} pDCs/mm²", yerr=yerr, capsize=6)

    plt.suptitle('Non-Linear Litifilimab Skin-to-Plasma AUC Ratio', fontsize=22, fontweight='bold', x=0.54)
    ax.set_title('Effects of IV Dose Size and pDC Density on Tissue Exposure', fontsize=18)
    ax.set_xlabel('IV Dose Size [mg/kg]', fontsize=18)
    ax.set_ylabel('AUC Ratio Skin vs Plasma [%]', fontsize=18)
    ax.legend(title='pDC Skin Density', title_fontsize=18, fontsize=16, loc='lower right', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticks(x + (bar_width * (len(patient_labels)-1) / 2))
    ax.set_xticklabels([f"{dose}" for dose in doses], fontsize=16) 

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 100)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:g}'))

    plt.tight_layout()
    save_plot(save_dir, "AUC_skin_plasma_ratios")

# Generate plot for Supplementary Figure 6
def plot_plasma_AUC(params, models, time_vector):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PK')
    os.makedirs(save_dir, exist_ok=True)

    patient_labels = ['1', '80', '400']
    doses = [0.05, 0.3, 1, 3, 10, 20, 40, 60]
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, 3))

    AUC_results = {label: {} for label in patient_labels}

    for label in patient_labels:
        model = models[f'SLE_model_{label}']
        acceptable_params = params[f'acceptable_params_{label}']
        final_params = params[f'final_params'][f'SLE_{label}']
        
        for dose in doses:
            sim = create_simulation_objects(model, 'SLE', bodyweight, custom_IV_doses=[dose])

            plasma_min, plasma_max = calculate_uncertainty(sim, time_vector, acceptable_params, 'SLE', 'PK_plasma_sim')
            plasma_best = simulate(final_params, sim, time_vector, 'PK_plasma_sim')

            AUC_results[label][dose] = {
                "best": np.trapezoid(plasma_best, time_vector),
                "min": np.trapezoid(plasma_min, time_vector),
                "max": np.trapezoid(plasma_max, time_vector)
            } 

    txt_path = os.path.join(save_dir, "AUC_plasma.txt")
    with open(txt_path, "w") as f:
        header = f"{'Density':<10} {'Dose':<6} {'AUC_best (µg*h/mL)':>20} {'AUC_min':>20} {'AUC_max':>20}\n"
        f.write(header)
        f.write("-" * len(header) + "\n")
        for label in patient_labels:
            for dose in doses:
                res = AUC_results[label][dose]
                f.write(f"{label:<10} {dose:<6} {res['best']:20.3f} {res['min']:20.3f} {res['max']:20.3f}\n")  
    
    fig, ax = plt.subplots(figsize=(10,8))
    bar_width = 0.25
    x = np.arange(len(doses))

    for i, label in enumerate(patient_labels):
        best = [AUC_results[label][dose]["best"] for dose in doses]
        min_val = [AUC_results[label][dose]["min"] for dose in doses]
        max_val = [AUC_results[label][dose]["max"] for dose in doses]

        yerr = [np.array(best) - np.array(min_val), np.array(max_val) - np.array(best)]
        ax.bar(x + i * bar_width, best, width=bar_width, color=colors[i], label=f"{label} pDCs/mm²", yerr=yerr, capsize=6)

    plt.suptitle('Absolute Litifilimab Plasma AUC', fontsize=22, fontweight='bold', x=0.54)
    ax.set_title('Impact of Skin pDC Density on Systemic Exposure', fontsize=18)
    ax.set_xlabel('IV Dose Size [mg/kg]', fontsize=18)
    ax.set_ylabel('Plasma AUC [µg·h/mL]', fontsize=18)
    ax.legend(title='pDC Skin Density', title_fontsize=18, fontsize=16, loc='upper left', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    ax.set_xticks(x + (bar_width * (len(patient_labels)-1) / 2))
    ax.set_xticklabels([f"{dose}" for dose in doses], fontsize=16) 

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')

    plt.tight_layout()
    save_plot(save_dir, "AUC_plasma")

# Generate plot for Figure 7B-C
def plot_skin_plasma_concentration_ratio(params, models, time_vector):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'PK')
    os.makedirs(save_dir, exist_ok=True)


    # Switch between plotting SLE (Figure 7B) and HV (Figure 7C) scenarios by commenting/uncommenting in the script:
    SLE_scenarios = {
        "Simulation (1 pDCs/mm² in Skin)": ('SLE_model_1', '1'),
        "Simulation (80 pDCs/mm² in Skin)": ('SLE_model_80', '80'),
        "Simulation (400 pDCs/mm² in Skin)": ('SLE_model_400', '400')
    }

    # HV_scenarios = {
    #     "Simulation (5100 pDCs/mL in Plasma)": ('HV_model', '80'),
    #     "Simulation (10200 pDCs/mL in Plasma)": ('HV_model_high', '80')
    # }
    
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, 3))
    linestyles = ['--', '-', ':']
    common_plasma_range = np.logspace(-4, 4, 500)
    start_index = np.searchsorted(time_vector, 12)

    fig, ax = plt.subplots(figsize=(10, 8))

    ref_x = np.logspace(-4, 4, 100) 
    ax.plot(ref_x, 0.157 * ref_x, 'k-', linewidth=3, label='Estimate for Non-Binding mAbs (15.7%)')
    ax.plot(ref_x, 0.0785 * ref_x, 'k--', linewidth=2, label='2-fold Error Margin')
    ax.plot(ref_x, 0.314 * ref_x, 'k--', linewidth=2)

    final_ratios = {}

    for i, (label, (m_key, p_suffix)) in enumerate(SLE_scenarios.items()):
    # for i, (label, (m_key, p_suffix)) in enumerate(HV_scenarios.items()):
        model = models[m_key]
        p_type = 'HV' if 'HV' in m_key else 'SLE'
        
        best_p = params['final_params']['HV' if p_type == 'HV' else f'SLE_{p_suffix}']
        acc_p = params[f'acceptable_params_{p_suffix}']

        sim_dict = create_simulation_objects(model, p_type, bodyweight, custom_IV_doses=[20])
        sim = list(sim_dict.values())[0]

        all_interp_skin = []
        for p in acc_p:
            adj_p = np.delete(np.array(p).copy(), [11, 16] if p_type == 'HV' else [10, 15])
            try:
                plasma = simulate(adj_p, sim, time_vector, 'PK_plasma_sim')[start_index:]
                skin = simulate(adj_p, sim, time_vector, 'PK_skin_sim')[start_index:]
                sort_idx = np.argsort(plasma)
                all_interp_skin.append(np.interp(common_plasma_range, plasma[sort_idx], skin[sort_idx], left=np.nan, right=np.nan))
            except RuntimeError: continue

        if all_interp_skin:
            ax.fill_between(common_plasma_range, np.nanmin(all_interp_skin, axis=0), np.nanmax(all_interp_skin, axis=0), 
                             color=colors[i], alpha=0.3)

        plasma_best = simulate(best_p, sim, time_vector, 'PK_plasma_sim')[start_index:]
        skin_best = simulate(best_p, sim, time_vector, 'PK_skin_sim')[start_index:]
        ax.plot(plasma_best, skin_best, label=label, color=colors[i], 
                linestyle=linestyles[i], linewidth=3)


        # Calculate the skin-plasma-ratio at either a given skin concentration or plasma concentration
        target_skin_conc = 1e-5
        # target_plasma_conc = 1e-4
        
        sort_idx = np.argsort(skin_best)
        # sort_idx = np.argsort(plasma_best)
        
        interpolated_plasma = np.interp(target_skin_conc, skin_best[sort_idx], plasma_best[sort_idx])
        # interpolated_skin = np.interp(target_plasma_conc, plasma_best[sort_idx], skin_best[sort_idx])
        
        ratio_percentage = (target_skin_conc / interpolated_plasma) * 100
        # ratio_percentage = (interpolated_skin / target_plasma_conc) * 100
        final_ratios[m_key] = ratio_percentage

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-4, 4e2); ax.set_ylim(1e-5, 4e2)
    ax.set_xlabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
    ax.set_ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize=18)

    ax.set_title('Divergence from Literature Estimate due to pDC-Driven TMDD', fontsize=18)
    # ax.set_title('Agreement with Literature Estimate >0.1 µg/ml in Plasma', fontsize=18)

    plt.suptitle('Biodistribution of Litifilimab in SLE/CLE Patients', fontsize=22, fontweight='bold', x=0.54)
    # plt.suptitle('Biodistribution of Litifilimab in Healthy Volunteers', fontsize=22, fontweight='bold', x=0.54)

    ax.legend(fontsize=16, loc='upper left'); ax.tick_params(axis='both', which='major', labelsize=16)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_plot(save_dir, "Skin_vs_plasma_distribution_SLE")
    # save_plot(save_dir, "Skin_vs_plasma_distribution_HV")

    print("\n--- Final Stabilized Concentration Ratios (Skin / Plasma) ---")
    print(f"400 pDC: {final_ratios.get('SLE_model_400', 0.0):.2f} %")
    print(f"80 pDC:  {final_ratios.get('SLE_model_80', 0.0):.2f} %")
    print(f"1 pDC:   {final_ratios.get('SLE_model_1', 0.0):.2f} %")
    # print(f"5100 pDCs/mL (HV Standard):  {final_ratios.get('HV_model', 0.0):.2f} %")
    # print(f"10200 pDCs/mL (HV High):    {final_ratios.get('HV_model_high', 0.0):.2f} %")
    print("-------------------------------------------------------------\n")

# Generate data for Figure 6B and Supplementary Figure 5
def simulate_loading_phase_clinical_menu(params, models, density_labels):
    save_dir = os.path.join(base_dir, 'Data')
    os.makedirs(save_dir, exist_ok=True)
    
    threshold = -90               
    loading_duration_w = 4.0
    loading_duration_days = 28.0
    
    absorption_buffer_days = 3.0 
    
    dose_settings = {
        '1': {'dose_vec': np.array([25, 50, 100, 150, 200, 250]), 'interval_days_vec': np.array([4, 7, 14, 21, 28]), 'interval_labels': ['Every 4 Days', 'Every Week', 'Every 2 Weeks', 'Every 3 Weeks', 'Every 4 Weeks']},
        '80': {'dose_vec': np.array([300, 350, 400, 450, 500, 600, 700]), 'interval_days_vec': np.array([4, 7, 10, 14]), 'interval_labels': ['Every 4 Days', 'Every Week', 'Every 10 Days', 'Every 2 Weeks']},
        '400': {'dose_vec': np.array([300, 400, 500, 600, 700, 800]), 'interval_days_vec': np.array([1, 2, 3, 4]), 'interval_labels': ['Every Day', 'Every 2 Days', 'Every 3 Days', 'Every 4 Days']}
    }

    for density_label in density_labels:
        if density_label not in dose_settings:
            continue
            
        dose_info = dose_settings[density_label]
        dose_vec = dose_info['dose_vec']
        interval_days_vec = dose_info['interval_days_vec']
        interval_labels = dose_info['interval_labels']

        surface_burden = np.full((len(dose_vec), len(interval_days_vec)), np.nan)
        
        best_p = params['final_params'][f'SLE_{density_label}']
        acc_p = params[f'acceptable_params_{density_label}']
        model = models[f'SLE_model_{density_label}']
        
        counter = 0
        total_cells = len(dose_vec) * len(interval_days_vec)
        
        for i, dose_mg in enumerate(dose_vec):
            for j in range(len(interval_days_vec) - 1, -1, -1):
                interval_days = interval_days_vec[j]
                counter += 1
                
                try:
                    sim_dict = create_simulation_objects(
                        model, 'SLE', bodyweight, 
                        custom_SC_doses=[{'size_mg': dose_mg, 'interval_weeks': interval_days / 7.0, 'total_weeks': loading_duration_w}]
                    )
                    sim = list(sim_dict.values())[0]
                    
                    target_time_days = loading_duration_days + absorption_buffer_days
                    t_vec = np.arange(0, int(target_time_days * 24.0) + 1, 1)
                    
                    y_best = simulate(best_p, sim, t_vec, 'PD_skin_sim')
                    
                    if y_best[-1] <= threshold:
                        y_min, y_max = calculate_uncertainty(sim, t_vec, acc_p, 'SLE', 'PD_skin_sim')
                        
                        if y_max[0] > -9999 and y_max[-1] <= threshold:
                            num_doses = int(np.ceil(loading_duration_days / interval_days))
                            surface_burden[i, j] = num_doses * dose_mg
                            
                            for k in range(j - 1, -1, -1):
                                shorter_interval = interval_days_vec[k]
                                auto_num_doses = int(np.ceil(loading_duration_days / shorter_interval))
                                surface_burden[i, k] = auto_num_doses * dose_mg
                                counter += 1
                            
                            break 
                        else:
                            surface_burden[i, j] = np.nan
                    else:
                        surface_burden[i, j] = np.nan
                            
                except RuntimeError:
                    surface_burden[i, j] = np.nan
                
                print(f"Simulating Loading Menu ({density_label} pDCs): Fast-tracked {counter}/{total_cells} regimens...")

        results = {
            "dose_vec": dose_vec.tolist(),
            "interval_days_vec": interval_days_vec.tolist(),
            "interval_labels": interval_labels,
            "surface_burden": surface_burden.tolist(),
            "loading_w": loading_duration_w
        }
        
        result_filename = os.path.join(save_dir, f"SLE_skin_SC_clinical_menu_loading_{density_label}.json")
        with open(result_filename, "w") as f:
            json.dump(results, f, indent=4)

# Generate plots for Figure 6B and Supplementary Figure 5
def plot_loading_phase_clinical_menu(params, models, density_labels):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'Dosing_protocols')
    os.makedirs(save_dir, exist_ok=True)
    
    fmt_int = lambda x: int(x) if x % 1 == 0 else x
    
    for density_label in density_labels:
        data_path = os.path.join(base_dir, 'Data', f"SLE_skin_SC_clinical_menu_loading_{density_label}.json")
        if not os.path.exists(data_path):
            continue

        with open(data_path, "r") as f:
            data_dict = json.load(f)
            
        dose_vec = np.array(data_dict['dose_vec'])
        interval_days_vec = np.array(data_dict['interval_days_vec'])
        interval_labels = data_dict['interval_labels']
        surface_burden = np.array(data_dict['surface_burden'])
        loading_w = data_dict['loading_w']

        data_matrix = surface_burden.T
        data_matrix = np.flipud(data_matrix)
        plot_interval_labels = interval_labels[::-1]
        
        cmap = plt.cm.get_cmap('RdYlGn_r').copy()
        cmap.set_bad('#e0e0e0')
        
        min_val = np.nanmin(data_matrix)
        max_val = np.nanmax(data_matrix)
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        min_idx_orig = np.unravel_index(np.nanargmin(surface_burden), surface_burden.shape)
        dose_idx_1, int_idx_1 = min_idx_orig[0], min_idx_orig[1]
        dose_1, int_1, int_label_1 = dose_vec[dose_idx_1], interval_days_vec[int_idx_1], interval_labels[int_idx_1]
        color_1 = cmap(norm(surface_burden[dose_idx_1, int_idx_1]))
        
        max_idx_orig = np.unravel_index(np.nanargmax(surface_burden), surface_burden.shape)
        dose_2, int_2, int_label_2 = dose_vec[max_idx_orig[0]], interval_days_vec[max_idx_orig[1]], interval_labels[max_idx_orig[1]]
        color_2 = cmap(norm(surface_burden[max_idx_orig[0], max_idx_orig[1]]))
        
        failed_indices = np.argwhere(np.isnan(surface_burden))
        if len(failed_indices) > 0:
            dose_f, int_f, int_label_f = dose_vec[failed_indices[-2][0]], interval_days_vec[failed_indices[-2][1]], interval_labels[failed_indices[-2][1]]
            color_f = '#888888'
        else:
            dose_f, int_f, int_label_f, color_f = None, None, None, None
            
        scenarios = [(dose_1, int_1, color_1, f"Most Efficient:\n{int(dose_1)} mg {int_label_1}")]
        if dose_1 != dose_2 or int_label_1 != int_label_2:
            scenarios.append((dose_2, int_2, color_2, f"Highest Burden:\n{int(dose_2)} mg {int_label_2}"))
        if dose_f is not None:
            scenarios.append((dose_f, int_f, color_f, f"Fails Target:\n{int(dose_f)} mg {int_label_f}"))
            
        N = len(scenarios)

        fig = plt.figure(figsize=(18, max(8, 2.5 * N))) 
        gs = fig.add_gridspec(N, 2, width_ratios=[1.2, 1], wspace=0.4, hspace=0.3)
        
        ax1 = fig.add_subplot(gs[:, 0])
        
        cax = ax1.imshow(data_matrix, cmap=cmap, aspect='auto', norm=norm)
        
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                val = data_matrix[i, j]
                if not np.isnan(val):
                    norm_val = norm(val)
                    text_color = "white" if (norm_val < 0.3 or norm_val > 0.7) else "black"
                    ax1.text(j, i, f"{int(val)}\nmg", ha="center", va="center", color=text_color, fontsize=13, fontweight='bold')
                else:
                    ax1.text(j, i, "Fails", ha="center", va="center", color="#888888", fontsize=13, fontstyle='italic')

        row_idx = len(interval_days_vec) - 1 - int_idx_1 
        rect = patches.Rectangle((dose_idx_1 - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=5, zorder=5, clip_on=False)
        ax1.add_patch(rect)

        ax1.set_xticks(np.arange(len(dose_vec)))
        ax1.set_xticklabels([f"{int(d)}" for d in dose_vec], fontsize=16)
        ax1.set_yticks(np.arange(len(plot_interval_labels)))
        ax1.set_yticklabels(plot_interval_labels, fontsize=16)
        
        ax1.set_xlabel('SC Dose Size [mg]', fontsize=18, labelpad=10)
        ax1.set_ylabel('Dosing Interval', fontsize=18, labelpad=10)
        ax1.set_title(f"Loading Protocol Selection Chart", fontsize=20)
        
        cbar = fig.colorbar(cax, ax=ax1, pad=0.03, aspect=20)
        cbar.set_label('Cumulative Litifilimab Dose [mg]', fontsize=18, labelpad=15)
        cbar.ax.tick_params(labelsize=16)
        
        ax1.set_xticks(np.arange(-.5, len(dose_vec), 1), minor=True)
        ax1.set_yticks(np.arange(-.5, len(plot_interval_labels), 1), minor=True)
        ax1.grid(which="minor", color="white", linestyle='-', linewidth=2)
        ax1.tick_params(which="minor", bottom=False, left=False)
        ax1.tick_params(which="major", bottom=False, left=False)
        for spine in ax1.spines.values(): spine.set_visible(False)

        model = models[f'SLE_model_{density_label}']
        best_p = params['final_params'][f'SLE_{density_label}']
        acc_p = params[f'acceptable_params_{density_label}']
        
        for i, (d, interval, c, lab) in enumerate(scenarios):
            ax_pd = fig.add_subplot(gs[i, 1])
            
            sim_dict = create_simulation_objects(model, 'SLE', bodyweight, 
                       custom_SC_doses=[{'size_mg': d, 'interval_weeks': interval / 7.0, 'total_weeks': loading_w}])
            sim = list(sim_dict.values())[0]
            t_vec = np.arange(0, int(loading_w * 168) + 1, 1)
            
            fill = None
            line = None

            try:
                y_min, y_max = calculate_uncertainty(sim, t_vec, acc_p, 'SLE', 'PD_skin_sim')
                y_best = simulate(best_p, sim, t_vec, 'PD_skin_sim')
                
                fill = ax_pd.fill_between(t_vec/168, y_min, y_max, color=c, alpha=0.3)
                line, = ax_pd.plot(t_vec/168, y_best, color=c, linewidth=2.5)
            except RuntimeError:
                pass 
        
            ax_pd.axhline(-90, color='k', linestyle='--', linewidth=2, alpha=0.8)
            
            ax_pd.set_xlim(0, loading_w)
            ax_pd.set_ylim(-105, 5)
            ax_pd.set_xticks(np.arange(0, int(loading_w) + 1, 1))
            
            ax_pd.spines['top'].set_visible(False)
            ax_pd.spines['right'].set_visible(False)
            ax_pd.tick_params(axis='y', labelsize=16)
       
            handles = []
            labels = []
            
            if fill is not None and line is not None:
                handles.append((fill, line))
                labels.append(lab)
            
            ax_pd.legend(handles, labels, loc='upper right', fontsize=14, framealpha=0.9)
            
            if i == 0:
                ax_pd.set_title(f"Simulation of Loading Protocols", fontsize=20)
            
            if i < N - 1:
                ax_pd.set_xticklabels([]) 
                ax_pd.spines['bottom'].set_visible(True)
            else:
                ax_pd.set_xlabel('Time [Weeks]', fontsize=18)
                ax_pd.tick_params(axis='x', labelsize=16)
            
            if i == N // 2:
                ax_pd.set_ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize=18, labelpad=15)
                ax_pd.text(loading_w * 0.5, -87, '10% Recovery Threshold', fontsize=14, va='bottom', ha='left', color='black')
        
        plt.suptitle(f"Patient-Specific Loading Protocol ({density_label} pDCs/mm²)", fontsize=24, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        save_plot(save_dir, f"Clinical_Menu_Loading_Burden_With_PD_{density_label}")
        plt.close(fig)

# Generate data for Figure 6B and Supplementary Figure 5
def simulate_minimum_maintenance_menu(params, models, density_labels):
    save_dir = os.path.join(base_dir, 'Data')
    os.makedirs(save_dir, exist_ok=True)
    
    threshold = -90               
    
    interval_settings = {
        '1': {'interval_days_vec': np.array([14, 28, 42, 56, 70]), 'interval_labels': ['Every 2 Weeks', 'Every 4 Weeks', 'Every 6 Weeks', 'Every 8 Weeks', 'Every 10 Weeks']},
        '80': {'interval_days_vec': np.array([3, 7, 14, 21, 28]), 'interval_labels': ['Every 3 Days', 'Every Week', 'Every 2 Weeks', 'Every 3 Weeks', 'Every 4 Weeks']},
        '400': {'interval_days_vec': np.array([1, 2, 4, 7]), 'interval_labels': ['Every Day', 'Every 2 Days', 'Every 4 Days', 'Every Week']}
    }

    dose_vec = np.arange(0, 1005, 5) 

    for density_label in density_labels:
        if density_label not in interval_settings:
            continue
            
        loading_data_path = os.path.join(base_dir, 'Data', f"SLE_skin_SC_clinical_menu_loading_{density_label}.json")
        try:
            with open(loading_data_path, "r") as f:
                load_data = json.load(f)
            surf_b = np.array(load_data['surface_burden'])
            min_idx = np.unravel_index(np.nanargmin(surf_b), surf_b.shape)
            
            load_dose = float(load_data['dose_vec'][min_idx[0]])
            load_int_days = float(load_data['interval_days_vec'][min_idx[1]])
            loading_w = float(load_data['loading_w'])
        except Exception as e:
            print(f"Could not load optimal protocol for {density_label}: {e}")
            continue

        loading_duration_h = float(loading_w * 168.0)
        loading_doses_count = int(np.ceil((loading_w * 7.0) / load_int_days))
        load_times_h = [float(c * load_int_days * 24.0) for c in range(loading_doses_count)]
            
        intervals = interval_settings[density_label]['interval_days_vec']
        labels = interval_settings[density_label]['interval_labels']
        
        required_doses = np.full(len(intervals), np.nan)
        maint_burdens = np.full(len(intervals), np.nan)
        
        best_p = params['final_params'][f'SLE_{density_label}']
        acc_p = params[f'acceptable_params_{density_label}']
        model = models[f'SLE_model_{density_label}']
        
        start_dose_idx = 0 
        
        for i, base_int in enumerate(intervals):
            base_int_h = float(base_int * 24.0)
            
            steady_state_days = 112.0
            num_maint_cycles = int(np.ceil(steady_state_days / base_int))
            num_maint_cycles = max(3, num_maint_cycles)
            
            maint_times_h = [float(loading_duration_h + c * base_int_h) for c in range(num_maint_cycles)]
            
            times_h = load_times_h + maint_times_h
            target_time_hours = maint_times_h[-1] + base_int_h
            t_vec = np.arange(0, int(target_time_hours) + 1, 1)
            
            found_dose = np.nan
            
            for d_idx in range(start_dose_idx, len(dose_vec)):
                dose_mg = float(dose_vec[d_idx])
                sizes_mg = [load_dose] * len(load_times_h) + [dose_mg] * len(maint_times_h)
                
                try:
                    sim_dict = create_simulation_objects(
                        model, 'SLE', bodyweight, 
                        custom_SC_doses=[{'custom_times_h': times_h, 'custom_sizes_mg': sizes_mg}]
                    )
                    sim = list(sim_dict.values())[0]
                    
                    y_best = simulate(best_p, sim, t_vec, 'PD_skin_sim')
                    
                    if y_best[-1] <= threshold:
                        y_min, y_max = calculate_uncertainty(sim, t_vec, acc_p, 'SLE', 'PD_skin_sim')
                        
                        if y_max[0] > -9999 and check_suppression_maintained(y_max, t_vec, threshold, loading_duration_h, target_time_hours):
                            found_dose = dose_mg
                            start_dose_idx = d_idx 
                            break 
                            
                except RuntimeError:
                    pass 
            
            if not np.isnan(found_dose):
                required_doses[i] = found_dose
                maintenance_duration_days = 140.0
                num_maint_doses_20w = int(np.ceil(maintenance_duration_days / base_int))
                maint_burdens[i] = found_dose * num_maint_doses_20w
            else:
                break
            
            print(f"Simulating Minimum Dose ({density_label} pDCs): Interval {base_int} days -> {found_dose} mg")

        results = {
            "base_intervals": intervals.tolist(),
            "interval_labels": labels,
            "required_doses": required_doses.tolist(),
            "maint_burdens": maint_burdens.tolist(),
            "loading_dose": load_dose,
            "loading_int_days": load_int_days,
            "loading_w": loading_w
        }
        
        result_filename = os.path.join(save_dir, f"SLE_skin_SC_min_maintenance_{density_label}.json")
        with open(result_filename, "w") as f:
            json.dump(results, f, indent=4)

# Generate plots for Figure 6B and Supplementary Figure 5
def plot_minimum_maintenance_menu(params, models, density_labels):
    save_dir = os.path.join(base_dir, 'Results', 'SLE', 'Dosing_protocols')
    os.makedirs(save_dir, exist_ok=True)
    
    for density_label in density_labels:
        data_path = os.path.join(base_dir, 'Data', f"SLE_skin_SC_min_maintenance_{density_label}.json")
        if not os.path.exists(data_path):
            continue

        with open(data_path, "r") as f:
            data_dict = json.load(f)
            
        labels = data_dict['interval_labels']
        intervals = np.array(data_dict['base_intervals'])
        doses = np.array(data_dict['required_doses'])
        burdens = np.array(data_dict['maint_burdens'])
        
        load_dose = data_dict['loading_dose']
        load_int_days = data_dict['loading_int_days']
        loading_w = data_dict['loading_w']
        
        valid_idx = ~np.isnan(doses)
        if not np.any(valid_idx): 
            continue
            
        labels = [labels[i] for i in range(len(labels)) if valid_idx[i]]
        intervals = intervals[valid_idx]
        doses = doses[valid_idx]
        burdens = burdens[valid_idx]

        N = len(intervals)
        
        fig = plt.figure(figsize=(14, max(6, 1.6 * N)))
        gs = fig.add_gridspec(N, 2, width_ratios=[1, 3.5], wspace=0.7, hspace=0.3)
        
        ax_left = fig.add_subplot(gs[:, 0])
        
        cmap = plt.cm.get_cmap('RdYlGn_r').copy()
        min_val, max_val = np.nanmin(burdens), np.nanmax(burdens)
        
        norm = mcolors.Normalize(vmin=min_val*0.9, vmax=max_val*1.1) if min_val == max_val else mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        burden_matrix = burdens.reshape(-1, 1)
        cax = ax_left.imshow(burden_matrix, cmap=cmap, aspect='auto', norm=norm)
        
        for i in range(N):
            norm_val = norm(burdens[i])
            text_color = "white" if (norm_val < 0.3 or norm_val > 0.7) else "black"
            ax_left.text(0, i, f"{int(doses[i])} mg", ha="center", va="center", color=text_color, fontsize=16, fontweight='bold')
            
        ax_left.set_xticks([])
        ax_left.set_yticks(np.arange(N))
        ax_left.set_yticklabels(labels, fontsize=16)
        ax_left.set_ylabel('Dosing Interval', fontsize=18, labelpad=10)
        ax_left.set_title("Minimum Required Dose", fontsize=20, pad=15)
        
        ax_left.set_yticks(np.arange(-.5, N, 1), minor=True)
        ax_left.grid(which="minor", color="white", linestyle='-', linewidth=3)
        ax_left.tick_params(which="both", bottom=False, left=False)
        
        for spine in ax_left.spines.values(): spine.set_visible(False)
        
        model = models[f'SLE_model_{density_label}']
        best_p = params['final_params'][f'SLE_{density_label}']
        acc_p = params[f'acceptable_params_{density_label}']
        
        loading_duration_h = float(loading_w * 168.0)
        loading_doses_count = int(np.ceil((loading_w * 7.0) / load_int_days))
        load_times_h = [float(c * load_int_days * 24.0) for c in range(loading_doses_count)]
        
        for i in range(N):
            ax_pd = fig.add_subplot(gs[i, 1])
            dose = doses[i]
            interval = intervals[i]
            
            base_int_h = float(interval * 24.0)
            
            target_weeks = 24.0
            maint_duration_days = (target_weeks - loading_w) * 7.0
            num_maint_cycles = int(np.ceil(maint_duration_days / interval))
            num_maint_cycles += 1
            
            maint_times_h = [float(loading_duration_h + c * base_int_h) for c in range(num_maint_cycles)]
            
            times_h = load_times_h + maint_times_h
            sizes_mg = [load_dose] * len(load_times_h) + [dose] * len(maint_times_h)
            
            target_time_hours = maint_times_h[-1] + base_int_h
            t_vec = np.arange(0, int(target_time_hours) + 1, 1)
            
            sim_dict = create_simulation_objects(
                model, 'SLE', bodyweight, 
                custom_SC_doses=[{'custom_times_h': times_h, 'custom_sizes_mg': sizes_mg}]
            )
            sim = list(sim_dict.values())[0]
            
            y_min, y_max = calculate_uncertainty(sim, t_vec, acc_p, 'SLE', 'PD_skin_sim')
            y_best = simulate(best_p, sim, t_vec, 'PD_skin_sim')
            
            t_plot = t_vec / 168.0 
            
            color = cmap(norm(burdens[i]))
            ax_pd.fill_between(t_plot, y_min, y_max, color=color, alpha=0.5)
            ax_pd.plot(t_plot, y_best, color=color, linewidth=3)
            
            ax_pd.axhline(-90, color='k', linestyle='--', linewidth=2, alpha=0.8)
            ax_pd.set_xlim(4, 24)
            ax_pd.set_ylim(-102, -85) 
            
            ax_pd.spines['top'].set_visible(False)
            ax_pd.spines['right'].set_visible(False)
            ax_pd.tick_params(axis='y', labelsize=16)
            
            ax_pd.set_xticks(np.arange(4, 25, 4))

            if i == 0:
                ax_pd.set_title("Simulation of Maintenance Protocols", fontsize=20, pad=15)
            
            if i < N - 1:
                ax_pd.set_xticklabels([])
                ax_pd.spines['bottom'].set_visible(True)
            else:
                ax_pd.set_xlabel('Time [Weeks]', fontsize=18)
                ax_pd.tick_params(axis='x', labelsize=16)
            
            if i == N // 2:
                ax_pd.set_ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize=18, labelpad=15)
        
        cbar = fig.colorbar(cax, ax=ax_left, pad=0.15, aspect=25)
        cbar.set_label('Cumulative Litifilimab Dose [mg]', fontsize=18, labelpad=15)
        cbar.ax.tick_params(labelsize=16)

        plt.suptitle(f"Patient-Specific Maintenance Protocol ({density_label} pDCs/mm²)", fontsize=24, fontweight='bold', x=0.52)
        
        plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.1) 
        save_plot(save_dir, f"Clinical_Menu_Min_Maintenance_{density_label}")
        plt.close(fig)


plot_IV_dose_response(params, models, time_vectors['IV'])

plot_skin_PK_simulations(params, models, data['SLE_CLE_PK_validation_data'], time_vectors)

plot_skin_PD_simulations(params, models, data['SLE_CLE_PK_validation_data'], time_vectors)

plot_skin_plasma_AUC_ratio(params, models, time_vectors['AUC'])

plot_plasma_AUC(params, models, time_vectors['AUC'])

plot_skin_plasma_concentration_ratio(params, models, time_vectors['Ratio'])

simulate_loading_phase_clinical_menu(params, models, density_labels)

plot_loading_phase_clinical_menu(params, models, density_labels)

simulate_minimum_maintenance_menu(params, models, density_labels)

plot_minimum_maintenance_menu(params, models, density_labels)