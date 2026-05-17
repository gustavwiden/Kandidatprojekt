# Importing the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
import json

from utils import base_dir, load_data, load_models, load_params, create_simulation_objects, calculate_uncertainty, simulate, save_plot, get_response_time

data = load_data('HV_PK_data', 'HV_PD_data', 'SLE_PK_data', 'SLE_PD_data', 'SLE_CLE_PK_validation_data', 'HV_vs_SLE_plasma_PK_response_data', 'HV_vs_SLE_plasma_PD_response_data')  

models = load_models('HV_model', 'SLE_model_80')    

params = load_params('final_params', 'acceptable_params_80')

# Overall average bodyweight for healthy volunteers (cohort 1-7) and SLE patients (cohort 8) in the phase 1 trial
bodyweight = 73

IV_doses = [0.05, 0.3, 1, 3, 10, 20]
SC_doses = [{'size_mg': 50, 'interval_weeks': None}]

HV_sims = create_simulation_objects(models['HV_model'], 'HV', bodyweight, dataset=data['HV_PK_data'])
SLE_sims = create_simulation_objects(models['SLE_model_80'], 'SLE', bodyweight, dataset=data['SLE_PK_data'], custom_IV_doses=IV_doses, custom_SC_doses=SC_doses)
SLE_CLE_sims = create_simulation_objects(models['SLE_model_80'], 'SLE', bodyweight, dataset=data['SLE_CLE_PK_validation_data'])


def plot_HV_PK_simulations(final_params, acceptable_params, sims, PK_data):
    save_dir = os.path.join(base_dir, 'Results', 'HV', 'PK')

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#fcf5ed')
    ax.set_facecolor('#fcf5ed')

    time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 3000, 1) for exp in PK_data}

    # Define colors and markers for each dose
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    
    # Loop through each dose
    for dose, color, marker in zip(PK_data.keys(), colors, markers):
        dose_times_weeks = np.array(PK_data[dose]['time']) / 168.0
        time_vector = time_vectors[dose]
        time_weeks = time_vector / 168.0

        # Calculate uncertainty range and best simulation
        y_min, y_max = calculate_uncertainty(sims[dose], time_vector, acceptable_params, 'HV', 'PK_sim')
        y_best = simulate(final_params, sims[dose], time_vector, 'PK_sim')

        # Plot uncertainty range and best simulation
        ax.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.3, label="Uncertainty")
        ax.plot(time_weeks, y_best, color=color, linewidth=2, label="Simulation")
        ax.errorbar(dose_times_weeks, PK_data[dose]['BIIB059_mean'], yerr=PK_data[dose]['SEM'], marker=marker, linestyle='None', markersize=6, color=color, capsize=3, label='Data')

    # plt.suptitle('PK Simulations in Plasma of Healthy Volunteer', fontsize=22, fontweight='bold', x=0.54)
    # ax.set_title("All Intravenous and Subcutaneous Doses from Phase 1 Trial", fontsize=18)
    ax.set_xlabel('Time [Weeks]', fontsize=20)
    ax.set_ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.set_yscale('log')
    ax.set_ylim(0.008, 1000)
    ax.set_xlim(-0.15, 22.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 

    plt.tight_layout()

    column_labels = ["0.05 mg/kg", "0.3 mg/kg", "1 mg/kg", "3 mg/kg", "10 mg/kg", "20 mg/kg", "50 mg"]
    row_labels = ["Uncertainty", "Simulation", "Data"]
    legend_handles = []

    # for i, label in enumerate(column_labels):
    #     color = colors[i]
    #     marker = markers[i]
    #     legend_handles.append(Patch(facecolor=color, alpha=0.3))
    #     legend_handles.append(Line2D([0], [0], color=color, lw=3))
    #     legend_handles.append(Line2D([0], [0], color=color, marker=marker, markersize=10))

    # ax = plt.gca()
    # legend = ax.legend(legend_handles, [''] * 21, ncol=7, loc='upper center', bbox_to_anchor=(0.7, 1), labelspacing=1.2, columnspacing=2, handletextpad=0.0,
    #                    title="IV 0.05 0.3 1.0 3.0 10 20    SC 50", title_fontsize=16, frameon=False)
    # legend._legend_box.align = "center"

    # for i, label in enumerate(row_labels):
    #     ax.text(0.45, 0.94 - (i * 0.05), label, transform=ax.transAxes, ha='right', fontsize=16, va='center')

    save_plot(save_dir, "HV_PK_plasma_simulations_poster")


def plot_HV_PD_simulations(final_params, acceptable_params, sims, PD_data, subset_doses=None):
    save_dir = os.path.join(base_dir, 'Results', 'HV', 'PD')
    time_vectors = {exp: np.arange(-400, PD_data[exp]["time"][-1] + 5000, 1) for exp in PD_data}

    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

    bar_labels, bar_colors = [], []
    best_times, fast_times, slow_times = [], [], []

    all_keys = list(PD_data.keys())

    for dose in all_keys:
        fig, ax = plt.subplots(figsize=(10, 8))

        if subset_doses and dose not in subset_doses:
            continue

        if 'SC' in dose:
            dose_size = dose.split('_')[1]
            label = f"SC {dose_size} mg"
        else:
            dose_size = dose.split('_')[1]
            if dose_size.startswith('0'):
                dose_size = dose_size[0] + '.' + dose_size[1:]
            label = f"IV {dose_size} mg/kg"

        idx = all_keys.index(dose)
        color = colors[idx]
        marker = markers[idx]

        time_vector = time_vectors[dose]
        time_weeks = time_vector / 168.0
        dose_times_weeks = np.array(PD_data[dose]['time']) / 168.0

        y_min, y_max = calculate_uncertainty(sims[dose], time_vector, acceptable_params, 'HV', 'PD_sim')
        y_best = simulate(final_params, sims[dose], time_vector, 'PD_sim')

        response_threshold = -60
        startpoint = 0

        best_times.append(get_response_time(y_best, response_threshold, startpoint, time_weeks, 'PD'))
        fast_times.append(get_response_time(y_max, response_threshold, startpoint, time_weeks, 'PD'))
        slow_times.append(get_response_time(y_min, response_threshold, startpoint, time_weeks, 'PD'))
        
        bar_labels.append(label)
        bar_colors.append(color)

        # Plot uncertainty range and best simulation
        ax.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.3, label="Uncertainty")
        ax.plot(time_weeks, y_best, color=color, linewidth=2, label="Simulation")
        ax.axhline(y=response_threshold, color='k', linestyle='--', alpha=0.8)

        if dose != 'IVdose_10_HV':
            ax.errorbar(dose_times_weeks, PD_data[dose]['BDCA2_median'], yerr=PD_data[dose]['SEM'], marker=marker, linestyle='None', markersize=8, color=color, capsize=4, elinewidth=2, label='Data')

        plt.suptitle('PD Simulation in Plasma of Healthy Volunteer', fontsize=22, fontweight='bold', x=0.54)
        ax.set_title(f'Subcutaneous Dose of {dose_size} mg' if dose == 'SCdose_50_HV' else f'Intravenous Dose of {dose_size} mg/kg', fontsize=18)
        ax.set_xlabel('Time [Weeks]', fontsize=18)
        ax.set_ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(-1.2, 42)
        ax.set_ylim(-118, 39)
        ax.legend(fontsize=16, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        save_plot(save_dir, f"HV_PD_plasma_simulation_{dose_size}")
    
    if bar_labels:
        fig_bar, ax_bar = plt.subplots(figsize=(9, max(2, len(bar_labels) * 0.6)), layout='constrained')

        best_arr = np.array(best_times, dtype=float)
        fast_arr = np.array(fast_times, dtype=float)
        slow_arr = np.array(slow_times, dtype=float)

        xerr = [best_arr - fast_arr, slow_arr - best_arr]
        y_pos = np.arange(len(bar_labels))

        ax_bar.barh(y_pos, best_arr, xerr=xerr, color=bar_colors, capsize=5, height=0.6, align='center', alpha=0.8)

        plt.suptitle('PD Response in Plasma', fontsize=22, fontweight='bold', x=0.58)
        ax_bar.set_title('Time to 25% Recovery from Maximal BDCA2 Suppression', fontsize=17)
        ax_bar.set_xlabel('Time [Weeks]', fontsize=18)
        ax_bar.tick_params(axis='both', which='major', labelsize=16)
        ax_bar.set_xlim(-1.2, 42)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(bar_labels, fontsize=16)
        ax_bar.invert_yaxis() 
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

        suffix = "subset" if subset_doses else "all_doses"
        save_plot(save_dir, f"HV_PD_plasma_response_{suffix}")


def plot_HV_PD_simulations_comparison(final_params, acceptable_params, sims, PD_data, selected_doses):
    save_dir = os.path.join(base_dir, 'Results', 'HV', 'PD')
    
    dose_settings = {
        'IVdose_005_HV': {'color': '#1b7837', 'marker': 'o', 'label': 'IV 0.05 mg/kg', 'size': '0.05'},
        'IVdose_03_HV':  {'color': '#01947b', 'marker': 's', 'label': 'IV 0.3 mg/kg',  'size': '0.3'},
        'IVdose_1_HV':   {'color': '#628759', 'marker': 'D', 'label': 'IV 1 mg/kg',    'size': '1.0'},
        'IVdose_3_HV':   {'color': '#70b5aa', 'marker': '^', 'label': 'IV 3 mg/kg',    'size': '3.0'},
        'IVdose_10_HV':  {'color': '#35978f', 'marker': 'v', 'label': 'IV 10 mg/kg',   'size': '10'},
        'IVdose_20_HV':  {'color': '#76b56e', 'marker': 'P', 'label': 'IV 20 mg/kg',   'size': '20'},
        'SCdose_50_HV':  {'color': '#6d65bf', 'marker': 'X', 'label': 'SC 50 mg',      'size': '50'}
    }

    time_vectors = {exp: np.arange(-400, PD_data[exp]["time"][-1] + 5000, 1) for exp in PD_data}

    bar_labels = []
    bar_colors = []
    best_times = []
    fast_times = []
    slow_times = []

    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained', facecolor='#fcf5ed')
    ax.set_facecolor('#fcf5ed')
    
    for dose in selected_doses:
        color = dose_settings[dose]['color']
        marker = dose_settings[dose]['marker']
        
        dose_times_weeks = np.array(PD_data[dose]['time']) / 168.0
        time_vector = time_vectors[dose]
        time_weeks = time_vector / 168.0

        y_min, y_max = calculate_uncertainty(sims[dose], time_vector, acceptable_params, 'HV', 'PD_sim')
        y_best = simulate(final_params, sims[dose], time_vector, 'PD_sim')

        response_threshold = -60
        startpoint = 0

        best_times.append(get_response_time(y_best, response_threshold, startpoint, time_weeks, 'PD'))
        fast_times.append(get_response_time(y_max, response_threshold, startpoint, time_weeks, 'PD'))
        slow_times.append(get_response_time(y_min, response_threshold, startpoint, time_weeks, 'PD'))
        
        bar_labels.append(dose_settings[dose]['label'])
        bar_colors.append(color)

        ax.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.3)
        ax.plot(time_weeks, y_best, color=color, linewidth=2)
        # ax.axhline(y=-60, color='k', linestyle='--', alpha=0.8, label='25% Recovery Threshold')
        # ax.text(18, -58, '25% Recovery Threshold', color='k', fontsize=16, va='bottom', ha='center')

        if dose != 'IVdose_10_HV':
            ax.errorbar(dose_times_weeks, PD_data[dose]['BDCA2_median'], yerr=PD_data[dose]['SEM'], 
                        marker=marker, linestyle='None', markersize=8, color=color, capsize=4, elinewidth=2)

    # plt.suptitle('PD Simulations in Plasma of Healthy Volunteer', fontsize=22, fontweight='bold')
    labels_for_title = [dose_settings[dose]['size'] for dose in selected_doses]
    # ax.set_title(f"Intravenous Doses of {' and '.join(f'{label} mg/kg' for label in labels_for_title)}", fontsize=18)
    ax.set_xlabel('Time [Weeks]', fontsize=20)
    ax.set_ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlim(-1.2, 42)
    ax.set_ylim(-118, 39)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend_handles = []
    # for dose in selected_doses:
    #     color = dose_settings[dose]['color']
    #     marker = dose_settings[dose]['marker']
    #     legend_handles.append(Patch(facecolor=color, alpha=0.3, edgecolor='none'))
    #     legend_handles.append(Line2D([0], [0], color=color, lw=3))
    #     legend_handles.append(Line2D([0], [0], color=color, marker=marker, linestyle='None', markersize=10))

    # num_doses = len(selected_doses)
    # blank_items = [''] * (num_doses * 3)
    # dynamic_legend_title = "   ".join([dose_settings[dose]['short'] for dose in selected_doses])

    # ax = plt.gca()
    # legend = ax.legend(legend_handles, blank_items, ncol=num_doses, loc='upper center', 
    #                    bbox_to_anchor=(0.9, 0.23), labelspacing=1.4, columnspacing=4.5, 
    #                    handletextpad=0.0, title=dynamic_legend_title, title_fontsize=16, frameon=False)
    # legend._legend_box.align = "center"

    # row_labels = ["Uncertainty", "Simulation", "Data"]
    # for i, label in enumerate(row_labels):
    #     ax.text(0.8, 0.14 - (i * 0.05), label, transform=ax.transAxes, ha='right', fontsize=16, va='center')
    
    dynamic_file_name = "_vs_".join(selected_doses)
    save_plot(save_dir, f"{dynamic_file_name}_PD_simulations_poster")

    fig_bar, ax_bar = plt.subplots(figsize=(10, len(selected_doses) * 0.8), layout='constrained')
    
    best_arr = np.array(best_times, dtype=float)
    fast_arr = np.array(fast_times, dtype=float)
    slow_arr = np.array(slow_times, dtype=float)

    xerr = [best_arr - fast_arr, slow_arr - best_arr]
    y_pos = np.arange(len(bar_labels))

    ax_bar.barh(y_pos, best_arr, xerr=xerr, color=bar_colors, capsize=5, height=0.6, align='center', alpha=0.8)

    # plt.suptitle('PD Response in Plasma', fontsize=22, fontweight='bold', x=0.54)
    # ax_bar.set_title("Time to 25% Recovery from Maximal BDCA2 Suppression", fontsize=17)
    ax_bar.set_xlabel('Time [Weeks]', fontsize=18)
    ax_bar.tick_params(axis='both', which='major', labelsize=16)
    ax_bar.set_xlim(-1.2, 42)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=16)
    ax_bar.invert_yaxis() 
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    save_plot(save_dir, f"{dynamic_file_name}_PD_plasma_response")


def plot_HV_vs_SLE_PK_simulations(HV_final_params, SLE_final_params, acceptable_params, HV_sims, SLE_sims, HV_PK_data, SLE_PK_data):
    save_dir = os.path.join(base_dir, 'Results', 'HV_vs_SLE', 'PK')

    response_results = {
        "HV": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 50], "Best": [], "Fast": [], "Slow": []},
        "SLE": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 50], "Best": [], "Fast": [], "Slow": []}
    }

    time_vectors = {exp: np.arange(-10, HV_PK_data[exp]["time"][-1] + 5000, 1) for exp in HV_PK_data}

    colors = plt.cm.Blues(np.linspace(0.7, 0.9, 2))
    labels = ["0.05 mg/kg IV Dose", "0.3 mg/kg IV Dose", "1 mg/kg IV Dose", "3 mg/kg IV Dose", "10 mg/kg IV Dose", "20 mg/kg IV Dose", "50 mg SC Dose"]

    for i, (HV_dose, label) in enumerate(zip(HV_sims.keys(), labels)):
        fig, ax = plt.subplots(figsize=(10, 8))

        SLE_dose = HV_dose.replace('HV', 'SLE')

        HV_dose_times_weeks = np.array(HV_PK_data[HV_dose]['time']) / 168.0
        time_vector = time_vectors[HV_dose]
        time_weeks = time_vector / 168.0

        HV_y_min, HV_y_max = calculate_uncertainty(HV_sims[HV_dose], time_vector, acceptable_params, 'HV', 'PK_sim')
        SLE_y_min, SLE_y_max = calculate_uncertainty(SLE_sims[SLE_dose], time_vector, acceptable_params, 'SLE', 'PK_sim')
        HV_y_best = simulate(HV_final_params, HV_sims[HV_dose], time_vector, 'PK_sim')
        SLE_y_best = simulate(SLE_final_params, SLE_sims[SLE_dose], time_vector, 'PK_sim')

        response_threshold = 1
        startpoint = 0.5 if HV_dose == 'SCdose_50_HV' else 0

        response_results["HV"]["Best"].append(get_response_time(HV_y_best, response_threshold, startpoint, time_weeks, 'PK'))
        response_results["HV"]["Fast"].append(get_response_time(HV_y_min, response_threshold, startpoint, time_weeks, 'PK'))
        response_results["HV"]["Slow"].append(get_response_time(HV_y_max, response_threshold, startpoint, time_weeks, 'PK'))
        
        response_results["SLE"]["Best"].append(get_response_time(SLE_y_best, response_threshold, startpoint, time_weeks, 'PK'))
        response_results["SLE"]["Fast"].append(get_response_time(SLE_y_min, response_threshold, startpoint, time_weeks, 'PK'))
        response_results["SLE"]["Slow"].append(get_response_time(SLE_y_max, response_threshold, startpoint, time_weeks, 'PK'))

        ax.fill_between(time_weeks, HV_y_min, HV_y_max, color=colors[0], alpha=0.3)
        ax.fill_between(time_weeks, SLE_y_min, SLE_y_max, color=colors[1], alpha=0.3)

        ax.plot(time_weeks, HV_y_best, color=colors[0], linewidth=2)
        ax.plot(time_weeks, SLE_y_best, color=colors[1], linewidth=2, linestyle='dashed')
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.8, label='1 µg/ml Threshold')
        # ax.text(7.5, 1.1, '1 µg/ml Threshold', color='k', fontsize=16, va='bottom', ha='center')

        ax.errorbar(HV_dose_times_weeks, HV_PK_data[HV_dose]['BIIB059_mean'], yerr=HV_PK_data[HV_dose]['SEM'], marker='o', linestyle='None', markersize=6, color=colors[0], capsize=3, label='HV_Data')

        if SLE_dose in SLE_PK_data:
            ax.errorbar(np.array(SLE_PK_data[SLE_dose]['time']) / 168.0, SLE_PK_data[SLE_dose]['BIIB059_mean'], yerr=SLE_PK_data[SLE_dose]['SEM'], marker='o', linestyle='None', markersize=6, color=colors[1], capsize=3, label='SLE_Data')

        plt.suptitle('PK Simulations in Plasma - HV vs SLE Patient', fontsize=22, fontweight='bold', x=0.54)
        ax.set_title(f'{label}', fontsize=18)
        ax.set_xlabel('Time [Weeks]', fontsize=20)
        ax.set_ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set_yscale('log')
        ax.set_ylim(0.001, 1000)
        ax.set_xlim(-0.15, 42)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # column_labels = ["HV", "SLE"] 
        # row_labels = ["Uncertainty", "Simulation", "Data"] 

        # legend_x = 0.85 if HV_dose in ['IVdose_005_HV', 'IVdose_03_HV', 'SCdose_50_HV'] else 0.35
        # text_x = 0.75 if HV_dose in ['IVdose_005_HV', 'IVdose_03_HV', 'SCdose_50_HV'] else 0.25      

        # legend_handles = []
        # for i, label in enumerate(column_labels):
        #     legend_handles.append(Patch(facecolor=colors[i], alpha=0.3, edgecolor='none'))
        #     legend_handles.append(Line2D([0], [0], color=colors[i], lw=3))
        #     legend_handles.append(Line2D([0], [0], color=colors[i], marker='o', linestyle='None', markersize=10))

        # ax = plt.gca()
        # legend = ax.legend(legend_handles, [''] * 21, ncol=2, loc='upper center', bbox_to_anchor=(legend_x, 0.30), labelspacing=1.4, 
        #                    columnspacing=3, handletextpad=0.0, title="HV    SLE", title_fontsize=16, frameon=False)
        # legend._legend_box.align = "center"

        # for i, label in enumerate(row_labels):
        #     ax.text(text_x, 0.20 - (i * 0.05), label, transform=ax.transAxes, ha='right', fontsize=16, va='center')

        save_plot(save_dir, f"PK_{HV_dose}_vs_SLE")

    fig_bar, ax_bar = plt.subplots(figsize=(10, 5), layout='constrained')
    
    best_HV = np.array(response_results['HV']['Best'], dtype=float)
    fast_HV = np.array(response_results['HV']['Fast'], dtype=float)
    slow_HV = np.array(response_results['HV']['Slow'], dtype=float)
    xerr_HV = [best_HV - fast_HV, slow_HV - best_HV]

    best_SLE = np.array(response_results['SLE']['Best'], dtype=float)
    fast_SLE = np.array(response_results['SLE']['Fast'], dtype=float)
    slow_SLE = np.array(response_results['SLE']['Slow'], dtype=float)
    xerr_SLE = [best_SLE - fast_SLE, slow_SLE - best_SLE]

    bar_labels = [f"IV {dose} mg/kg" if dose != 50 else "SC 50 mg" for dose in response_results['HV']['Dose']]
    y_pos = np.arange(len(bar_labels))

    ax_bar.barh(y_pos - 0.2, best_HV, xerr=xerr_HV, color=colors[0], capsize=5, height=0.4, align='center', alpha=0.8, label='HV')
    ax_bar.barh(y_pos + 0.2, best_SLE, xerr=xerr_SLE, color=colors[1], capsize=5, height=0.4, align='center', alpha=0.8, label='SLE')

    plt.suptitle('PK Response in Plasma - HV vs SLE Patient', fontsize=22, fontweight='bold', x=0.58)
    ax_bar.set_title('Time with Litifilimab Concentration >1 µg/ml', fontsize=17)
    
    ax_bar.set_xlabel('Time [Weeks]', fontsize=18)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=16)
    ax_bar.invert_yaxis()
    
    ax_bar.set_xlim(-1.2, 42)
    ax_bar.tick_params(axis='both', which='major', labelsize=16)
    ax_bar.legend(fontsize=16, loc='upper right', frameon=True)
    
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    save_plot(save_dir, "HV_vs_SLE_plasma_PK_response")

    # data_save_path = os.path.join(base_dir, 'Data', 'HV_vs_SLE_plasma_PK_response_data.json')
    # with open(data_save_path, "w") as f:
    #     json.dump(response_results, f, indent=4)


def plot_HV_vs_SLE_PD_simulations(HV_final_params, SLE_final_params, acceptable_params, HV_sims, SLE_sims, HV_PD_data, SLE_PD_data):
    save_dir = os.path.join(base_dir, 'Results', 'HV_vs_SLE', 'PD')

    response_results = {
        "HV": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 50], "Best": [], "Fast": [], "Slow": []},
        "SLE": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 50], "Best": [], "Fast": [], "Slow": []}
    }

    time_vectors = {exp: np.arange(-400, HV_PD_data[exp]["time"][-1] + 5000, 1) for exp in HV_PD_data}

    colors = plt.cm.Reds(np.linspace(0.7, 0.9, 2))
    labels = ["0.05 mg/kg IV Dose", "0.3 mg/kg IV Dose", "1 mg/kg IV Dose", "3 mg/kg IV Dose", "10 mg/kg IV Dose", "20 mg/kg IV Dose", "50 mg SC Dose"]

    for i, (HV_dose, label) in enumerate(zip(HV_sims.keys(), labels)):
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#fcf5ed')
        ax.set_facecolor('#fcf5ed')

        SLE_dose = HV_dose.replace('HV', 'SLE')

        HV_dose_times_weeks = np.array(HV_PD_data[HV_dose]['time']) / 168.0
        time_vector = time_vectors[HV_dose]
        time_weeks = time_vector / 168.0

        HV_y_min, HV_y_max = calculate_uncertainty(HV_sims[HV_dose], time_vector, acceptable_params, 'HV', 'PD_sim')
        SLE_y_min, SLE_y_max = calculate_uncertainty(SLE_sims[SLE_dose], time_vector, acceptable_params, 'SLE', 'PD_sim')
        HV_y_best = simulate(HV_final_params, HV_sims[HV_dose], time_vector, 'PD_sim')
        SLE_y_best = simulate(SLE_final_params, SLE_sims[SLE_dose], time_vector, 'PD_sim')

        response_threshold = -60
        startpoint = 0

        response_results["HV"]["Best"].append(get_response_time(HV_y_best, response_threshold, startpoint, time_weeks, 'PD'))
        response_results["HV"]["Fast"].append(get_response_time(HV_y_max, response_threshold, startpoint, time_weeks, 'PD'))
        response_results["HV"]["Slow"].append(get_response_time(HV_y_min, response_threshold, startpoint, time_weeks, 'PD'))
        
        response_results["SLE"]["Best"].append(get_response_time(SLE_y_best, response_threshold, startpoint, time_weeks, 'PD'))
        response_results["SLE"]["Fast"].append(get_response_time(SLE_y_max, response_threshold, startpoint, time_weeks, 'PD'))
        response_results["SLE"]["Slow"].append(get_response_time(SLE_y_min, response_threshold, startpoint, time_weeks, 'PD'))

        ax.fill_between(time_weeks, HV_y_min, HV_y_max, color=colors[0], alpha=0.3)
        ax.fill_between(time_weeks, SLE_y_min, SLE_y_max, color=colors[1], alpha=0.3)

        ax.plot(time_weeks, HV_y_best, color=colors[0], linewidth=2)
        ax.plot(time_weeks, SLE_y_best, color=colors[1], linewidth=2, linestyle='dashed')
        ax.axhline(y=response_threshold, color='k', linestyle='--', alpha=0.8, label='25% Recovery Threshold')
        # ax.text(30, -58, '25% Recovery Threshold', color='k', fontsize=16, va='bottom', ha='center')

        if HV_dose != 'IVdose_10_HV':
            ax.errorbar(HV_dose_times_weeks, HV_PD_data[HV_dose]['BDCA2_median'], yerr=HV_PD_data[HV_dose]['SEM'], marker='o', linestyle='None', markersize=6, color=colors[0], capsize=3, label='HV_Data')

        if SLE_dose in SLE_PD_data:
            ax.errorbar(np.array(SLE_PD_data[SLE_dose]['time']) / 168.0, SLE_PD_data[SLE_dose]['BDCA2_median'], yerr=SLE_PD_data[SLE_dose]['SEM'], marker='o', linestyle='None', markersize=6, color=colors[1], capsize=3, label='SLE_Data')

        # plt.suptitle('PD Simulations in Plasma - HV vs SLE Patient', fontsize=22, fontweight='bold', x=0.52)
        # ax.set_title(f'{label}', fontsize=18)
        ax.set_xlabel('Time [Weeks]', fontsize=20)
        ax.set_ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax.set_xlim(-1.2, 42)
        ax.set_ylim(-118, 39)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)  

        plt.tight_layout()

        # column_labels = ["HV", "SLE"] 
        # row_labels = ["Uncertainty", "Simulation", "Data"]      

        # legend_handles = []
        # for i, label in enumerate(column_labels):
        #     legend_handles.append(Patch(facecolor=colors[i], alpha=0.3, edgecolor='none'))
        #     legend_handles.append(Line2D([0], [0], color=colors[i], lw=3))
        #     legend_handles.append(Line2D([0], [0], color=colors[i], marker='o', linestyle='None', markersize=10))

        # ax = plt.gca()
        # legend = ax.legend(legend_handles, [''] * 21, ncol=2, loc='upper center', bbox_to_anchor=(0.9, 0.24), labelspacing=1.4, 
        #                    columnspacing=3, handletextpad=0.0, title="HV    SLE", title_fontsize=16, frameon=False)
        # legend._legend_box.align = "center"

        # for i, label in enumerate(row_labels):
        #     ax.text(0.8, 0.14 - (i * 0.05), label, transform=ax.transAxes, ha='right', fontsize=16, va='center')

        save_plot(save_dir, f"PD_{HV_dose}_vs_SLE_poster")

    fig_bar, ax_bar = plt.subplots(figsize=(10, 5), layout='constrained')
    
    best_HV = np.array(response_results['HV']['Best'], dtype=float)
    fast_HV = np.array(response_results['HV']['Fast'], dtype=float)
    slow_HV = np.array(response_results['HV']['Slow'], dtype=float)
    xerr_HV = [best_HV - fast_HV, slow_HV - best_HV]

    best_SLE = np.array(response_results['SLE']['Best'], dtype=float)
    fast_SLE = np.array(response_results['SLE']['Fast'], dtype=float)
    slow_SLE = np.array(response_results['SLE']['Slow'], dtype=float)
    xerr_SLE = [best_SLE - fast_SLE, slow_SLE - best_SLE]

    bar_labels = [f"IV {dose} mg/kg" if dose != 50 else "SC 50 mg" for dose in response_results['HV']['Dose']]
    y_pos = np.arange(len(bar_labels))

    ax_bar.barh(y_pos - 0.2, best_HV, xerr=xerr_HV, color=colors[0], capsize=5, height=0.4, align='center', alpha=0.8, label='HV')
    ax_bar.barh(y_pos + 0.2, best_SLE, xerr=xerr_SLE, color=colors[1], capsize=5, height=0.4, align='center', alpha=0.8, label='SLE')

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=16)
    ax_bar.invert_yaxis() 

    ax_bar.set_xlabel('Time [Weeks]', fontsize=18)
    ax_bar.set_title('Time to 25% Recovery from Maximal BDCA2 Suppression', fontsize=17)
    plt.suptitle('PD Response in Plasma - HV vs SLE Patient', fontsize=22, fontweight='bold', x=0.58)
    
    ax_bar.set_xlim(-1.2, 42)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.tick_params(axis='both', which='major', labelsize=16)
    ax_bar.legend(fontsize=16, loc='upper right', frameon=True)

    save_plot(save_dir, "HV_vs_SLE_plasma_PD_response")

    # data_save_path = os.path.join(base_dir, 'Data', 'HV_vs_SLE_plasma_PD_response_data.json')
    # with open(data_save_path, "w") as f:
    #     json.dump(response_results, f, indent=4)


def plot_SLE_PK_validation_simulations(final_params, acceptable_params, sims, PK_data):
    save_dir = os.path.join(base_dir, 'Results', 'Validation')
    time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 400, 1) for exp in PK_data}

    dose_settings = {
        'SCdose_50_SLE':  {'size': '50',  'patient': 'SLE'},
        'SCdose_150_SLE': {'size': '150', 'patient': 'SLE'},
        'SCdose_450_SLE': {'size': '450', 'patient': 'SLE'},
        'SCdose_50_CLE':  {'size': '50',  'patient': 'CLE'},
        'SCdose_150_CLE': {'size': '150', 'patient': 'CLE'},
        'SCdose_450_CLE': {'size': '450', 'patient': 'CLE'},
    }
    
    color = '#6d65bf'
    marker = 'X'

    for i, dose in enumerate(PK_data.keys()):
        fig,ax = plt.subplots(figsize=(10, 8), facecolor='#fcf5ed')
        ax.set_facecolor('#fcf5ed')
        patient = dose_settings[dose]['patient']
        dose_size = dose_settings[dose]['size']
        
        dose_times_weeks = np.array(PK_data[dose]['time']) / 168.0
        time_vector = time_vectors[dose]
        time_weeks = time_vector / 168.0

        y_min, y_max = calculate_uncertainty(sims[dose], time_vector, acceptable_params, 'SLE', 'PK_sim')
        y_best = simulate(final_params, sims[dose], time_vector, 'PK_sim')

        ax.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.3, label="Uncertainty")
        ax.plot(time_weeks, y_best, color=color, linewidth=3, label="Simulation")
        ax.errorbar(dose_times_weeks, PK_data[dose]['BIIB059_mean'], yerr=PK_data[dose]['SEM'], marker=marker, linestyle='None', elinewidth=2, markersize=8, color=color, capsize=4, label='Validation Data')

        # plt.suptitle(f'PK Validation in Plasma of {patient} Patient', fontsize=22, fontweight='bold', x=0.54)
        # if patient == 'SLE':
        #     ax.set_title(f'Phase 2: {dose_size} mg SC (W0, 2, 4 + Q4W to W20)', fontsize=18)
        # else:
        #     ax.set_title(f'Phase 2: {dose_size} mg SC (W0, 2, 4 + Q4W to W12)', fontsize=18)
        ax.set_xlabel('Time [weeks]', fontsize=20)
        ax.set_ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=20, loc='upper right', facecolor='#fcf5ed', frameon=True)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        save_plot(save_dir, f"{patient}_PK_validation_{dose}_poster")


# plot_HV_PK_simulations(params['final_params']['HV'], params['acceptable_params_80'], HV_sims, data['HV_PK_data'])

# plot_HV_PD_simulations(params['final_params']['HV'], params['acceptable_params_80'], HV_sims, data['HV_PD_data'])

# plot_HV_PD_simulations_comparison(params['final_params']['HV'], params['acceptable_params_80'], HV_sims, data['HV_PD_data'], selected_doses = ['IVdose_03_HV', 'IVdose_20_HV'])

plot_HV_vs_SLE_PK_simulations(params['final_params']['HV'], params['final_params']['SLE_80'], params['acceptable_params_80'], HV_sims, SLE_sims, data['HV_PK_data'], data['SLE_PK_data'])

# plot_HV_vs_SLE_PD_simulations(params['final_params']['HV'], params['final_params']['SLE_80'], params['acceptable_params_80'], HV_sims, SLE_sims, data['HV_PD_data'], data['SLE_PD_data'])

# plot_SLE_PK_validation_simulations(params['final_params']['SLE_80'], params['acceptable_params_80'], SLE_CLE_sims, data['SLE_CLE_PK_validation_data'])
