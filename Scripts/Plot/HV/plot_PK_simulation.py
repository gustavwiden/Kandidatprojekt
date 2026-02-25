# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sund
import json

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_model.txt file and read its contents
with open("../../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the mPBPK_SLE_model_80_pdc_mm2.txt file and read its contents
with open("../../../Models/mPBPK_SLE_model_80_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

# Load PK data
with open("../../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load SLE PK data
with open("../../../Data/SLE_PK_data.json", "r") as f:
    SLE_PK_data = json.load(f)

# Load acceptable parameters for mPBPK_model from PL
acceptable_params = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL_80_pdc_mm2.csv", delimiter=",").tolist()

# Load best parameter set for mPBPK_model_80_pdc_mm2
with open("../../../Models/final_parameters_SLE.json", "r") as f:
    final_params_SLE = json.load(f)

with open("../../../Models/final_parameters.json", "r") as f:
    final_params = json.load(f)

# Define a function to plot all doses with uncertainty in the same figure
def plot_all_doses_with_uncertainty(final_params, acceptable_params, sims, PK_data, time_vectors, save_dir='../../../Results/HV/PK'):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors and markers for each dose
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']
    
    # Loop through each experiment
    for i, (experiment, color) in enumerate(zip(PK_data.keys(), colors)):
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params:
            HV_params = np.delete(params.copy(), [11,16])
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=HV_params, reset=True)
                y_sim = sims[experiment].feature_data[:, 0]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range (x in weeks)
        time_weeks = timepoints / 168.0
        ax.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.3, label="Uncertainty")

        # Plot selected parameter set (simulate using hours, plot using weeks)
        sims[experiment].simulate(time_vector=timepoints, parameter_values=final_params, reset=True)
        y_selected = sims[experiment].feature_data[:, 0]
        ax.plot(time_weeks, y_selected, color=color, linewidth=2, label="Simulation")

        # Plot experimental data (convert times to weeks)
        marker = markers[i]
        exp_times_weeks = np.array(PK_data[experiment]['time']) / 168.0
        ax.errorbar(
            exp_times_weeks,
            PK_data[experiment]['BIIB059_mean'],
            yerr=PK_data[experiment]['SEM'],
            fmt=marker,
            markersize=6,
            color=color,
            linestyle='None',
            capsize=3,
            label='Data'
        )

    hours_per_week = 168.0

    ax.set_xlabel('Time [Weeks]', fontsize=18)
    ax.set_ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
    ax.set_title("Intravenous and Subcutaneous Doses in [mg/kg] and [mg]", fontsize=18)
    plt.suptitle('PK Simulations in Plasma of Healthy Volunteers', fontsize=22, fontweight='bold', x=0.54)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yscale('log')
    ax.set_ylim(0.005, 1000)
    # convert previous xlim from hours to weeks
    ax.set_xlim(-25.0 / hours_per_week, 2750.0 / hours_per_week)
    ax.tick_params(axis='both', which='major', labelsize=16)   
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) 

    column_labels = ["0.05 mg/kg", "0.3 mg/kg", "1 mg/kg", "3 mg/kg", "10 mg/kg", "20 mg/kg", "50 mg"]

    legend_handles = []
    for i, label in enumerate(column_labels):
        color = colors[i]
        marker = markers[i]
        legend_handles.append(Patch(facecolor=color, alpha=0.3, edgecolor='none'))
        legend_handles.append(Line2D([0], [0], color=color, lw=3))
        legend_handles.append(Line2D([0], [0], color=color, marker=marker, 
                                     linestyle='None', markersize=10))

    ax = plt.gca()
    legend = ax.legend(
        legend_handles, 
        [''] * 21, 
        ncol=7, 
        loc='upper center', 
        bbox_to_anchor=(0.52, -0.14),
        labelspacing=0.8,
        columnspacing=4.5,
        handletextpad=0.0,
        title="IV 0.05    IV 0.3    IV 1.0    IV 3.0    IV 10    IV 20    SC 50", 
        title_fontsize=16,
        frameon=False
    )
    legend._legend_box.align = "center"

    row_labels = ["Uncertainty", "Simulation", "Data"]
    for i, label in enumerate(row_labels):
        ax.text(0.14, -0.24 - (i * 0.05), label, # Adjusted offsets
                transform=ax.transAxes, ha='right', fontsize=16, va='center')

    # Save the figure
    # save_path = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.png")
    # plt.savefig(save_path, format='png', dpi=600)
    # plt.close()

    save_path_svg = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.svg")
    plt.savefig(save_path_svg, format='svg')
    plt.close()


def plot_HV_vs_SLE_PK_simulation(final_params, final_params_SLE, acceptable_params, sims_HV, sims_SLE, PK_data_HV, SLE_PK_data, time_vectors, save_dir='../../../Results/SLE/PK'):
    os.makedirs(save_dir, exist_ok=True)

    colors = plt.cm.Blues(np.linspace(0.7, 0.9, 2))
    labels = ["0.05 mg/kg IV Dose", "0.3 mg/kg IV Dose", "1 mg/kg IV Dose", "3 mg/kg IV Dose", "10 mg/kg IV Dose", "20 mg/kg IV Dose", "50 mg SC Dose"]

    for i, (experiment, label) in enumerate(zip(PK_data_HV.keys(), labels)):
        fig, ax = plt.subplots(figsize=(10, 8))
        timepoints = time_vectors[experiment]

        y_min_HV = np.full_like(timepoints, 10000)
        y_max_HV = np.full_like(timepoints, -10000)
        y_min_SLE = np.full_like(timepoints, 10000)
        y_max_SLE = np.full_like(timepoints, -10000)

        for params in acceptable_params:
            HV_params = np.delete(params.copy(), [11,16])
            SLE_params = np.delete(params.copy(), [10,15])
            try:
                sims_HV[experiment].simulate(time_vector=timepoints, parameter_values=HV_params, reset=True)
                y_sim_HV = sims_HV[experiment].feature_data[:, 0]
                y_min_HV = np.minimum(y_min_HV, y_sim_HV)
                y_max_HV = np.maximum(y_max_HV, y_sim_HV)

                sims_SLE[experiment].simulate(time_vector=timepoints, parameter_values=SLE_params, reset=True)
                y_sim_SLE = sims_SLE[experiment].feature_data[:, 0]
                y_min_SLE = np.minimum(y_min_SLE, y_sim_SLE)
                y_max_SLE = np.maximum(y_max_SLE, y_sim_SLE)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e
        hours_per_week = 168.0
        time_weeks = timepoints / hours_per_week

        ax.fill_between(time_weeks, y_min_HV, y_max_HV, color=colors[0], alpha=0.3)
        ax.fill_between(time_weeks, y_min_SLE, y_max_SLE, color=colors[1], alpha=0.3)

        sims_HV[experiment].simulate(time_vector=timepoints, parameter_values=final_params, reset=True)
        sims_SLE[experiment].simulate(time_vector=timepoints, parameter_values=final_params_SLE, reset=True)

        ax.plot(time_weeks, sims_HV[experiment].feature_data[:, 0], color=colors[0], linewidth=2)
        ax.plot(time_weeks, sims_SLE[experiment].feature_data[:, 0], color=colors[1], linewidth=2, linestyle='dashed')

        exp_times_weeks = np.array(PK_data[experiment]['time']) / hours_per_week
        ax.errorbar(
            exp_times_weeks,
            PK_data[experiment]['BIIB059_mean'],
            yerr=PK_data[experiment]['SEM'],
            fmt='o',
            markersize=6,
            color=colors[0],
            linestyle='None',
            capsize=3,
            label='HV Data'
        )

        if experiment == 'IVdose_20_HV':
            ax.errorbar(
                np.array(SLE_PK_data['IVdose_20_SLE']['time']) / hours_per_week,
                SLE_PK_data['IVdose_20_SLE']['BIIB059_mean'],
                yerr=SLE_PK_data['IVdose_20_SLE']['SEM'],
                fmt='o',
                markersize=6,
                color=colors[1],
                linestyle='None',
                capsize=3,
                label='SLE Data'
            )

        ax.set_xlabel('Time [Weeks]', fontsize=18)
        ax.set_ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
        ax.set_title("Healthy Volunteer vs SLE Patient", fontsize=18)
        plt.suptitle(f'PK Simulations in Plasma for a {label}', fontsize=22, fontweight='bold', x=0.54)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yscale('log')
        # ax.set_ylim(0.005, 1000)
        #ax.set_xlim(-25.0 / hours_per_week, 2750.0 / hours_per_week)
        ax.tick_params(axis='both', which='major', labelsize=16)   
        plt.tight_layout()

        column_labels = ["HV", "SLE"]

        legend_handles = []
        for i, label in enumerate(column_labels):
            color = colors[i]
            marker = 'o'
            legend_handles.append(Patch(facecolor=color, alpha=0.3, edgecolor='none'))
            legend_handles.append(Line2D([0], [0], color=color, lw=3))
            legend_handles.append(Line2D([0], [0], color=color, marker=marker, 
                                        linestyle='None', markersize=10))

        ax = plt.gca()
        legend = ax.legend(
            legend_handles, 
            [''] * 21, 
            ncol=2, 
            loc='upper center', 
            bbox_to_anchor=(0.38, 0.24),
            labelspacing=1.4,
            columnspacing=3,
            handletextpad=0.0,
            title="HV    SLE", 
            title_fontsize=16,
            frameon=False
        )
        legend._legend_box.align = "center"

        row_labels = ["Uncertainty", "Simulation", "Data"]
        for i, label in enumerate(row_labels):
            ax.text(0.28, 0.14 - (i * 0.05), label, # Adjusted offsets
                    transform=ax.transAxes, ha='right', fontsize=16, va='center')

        # Save the figure
        save_path_svg = os.path.join(save_dir, f"PK_HV_vs_SLE_{experiment}.svg")
        plt.savefig(save_path_svg, format='svg')
        plt.close()


# Install the model
sund.install_model('../../../Models/mPBPK_model.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_80_pdc_mm2.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_model")
model_SLE = sund.load_model("mPBPK_SLE_model_80_pdc_mm2")

# Average bodyweight for healthy volunteers (HV) (cohort 1-7 in the phase 1 trial)
bodyweight = 73

# Creating activity objects for each dose
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output('piecewise_constant', "IV_in",  t = PK_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output('piecewise_constant', "IV_in",  t = PK_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output('piecewise_constant', "IV_in",  t = PK_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output('piecewise_constant', "IV_in",  t = PK_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output('piecewise_constant', "IV_in",  t = PK_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output('piecewise_constant', "IV_in",  t = PK_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output('piecewise_constant', "SC_in",  t = PK_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data['SCdose_50_HV']['input']['SC_in']['f'])

# Creating simulation objects for each dose
model_sims_HV = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

model_sims_SLE = {
    'IVdose_005_HV': sund.Simulation(models = model_SLE, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model_SLE, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model_SLE, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model_SLE, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model_SLE, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model_SLE, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model_SLE, activities = SC_50_HV, time_unit = 'h')
}

# Define time vectors for each dose
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

# Plot all doses with uncertainty
plot_all_doses_with_uncertainty(final_params, acceptable_params, model_sims_HV, PK_data, time_vectors)

# Plot HV vs SLE PK simulation
plot_HV_vs_SLE_PK_simulation(final_params, final_params_SLE, acceptable_params, model_sims_HV, model_sims_SLE, PK_data, SLE_PK_data, time_vectors)
