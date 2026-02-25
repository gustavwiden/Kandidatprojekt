# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
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

# Open the PD data file and read its contents
with open("../../../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

with open("../../../Data/SLE_PD_data.json", "r") as f:
    SLE_PD_data = json.load(f)

# Load acceptable parameters for mPBPK-model from PL
acceptable_params = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL_80_pdc_mm2.csv", delimiter=",").tolist()

# Load final parameters for mPBPK-model form optimization
with open("../../../Models/final_parameters.json", "r") as f:
    final_params = json.load(f)

with open("../../../Models/final_parameters_SLE.json", "r") as f:
    final_params_SLE = json.load(f)

# Define a function to plot all doses with uncertainty in different figures
def plot_all_doses_with_uncertainty(selected_params, acceptable_params, sims, PD_data, time_vectors, save_dir='../../../Results/HV/PD'):
    os.makedirs(save_dir, exist_ok=True)

    # Define colors and markers for each dose
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'P', 'X']
    doses = ['0.05 mg/kg IV Dose', '0.3 mg/kg IV Dose', '1 mg/kg IV Dose', '3 mg/kg IV Dose', '20 mg/kg IV Dose', '50 mg SC Dose']

    #Loop through each experiment
    for i, (experiment, color, dose) in enumerate(zip(PD_data.keys(), colors, doses)):
        plt.figure(figsize=(10, 8))
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params:
            HV_params = np.delete(params.copy(), [11,16])
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=HV_params, reset=True)
                y_sim = sims[experiment].feature_data[:, 1]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Convert to weeks for plotting (1 week = 168 hours)
        hours_per_week = 168.0
        timepoints_weeks = timepoints / hours_per_week
        
        # Plot uncertainty range
        plt.fill_between(timepoints_weeks, y_min, y_max, color=color, alpha=0.3, label='Uncertainty')

        # Plot selected parameter set
        sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        y_selected = sims[experiment].feature_data[:, 1]
        plt.plot(timepoints_weeks, y_selected, color=color, label='Simulation', linewidth=3)

        # Plot experimental data
        marker = markers[i]
        exp_times_weeks = np.array(PD_data[experiment]['time']) / hours_per_week
        plt.errorbar(
            exp_times_weeks,
            PD_data[experiment]['BDCA2_median'],
            yerr=PD_data[experiment]['SEM'],
            marker=marker,
            markersize=8,
            color=color,
            linestyle='None',
            capsize=4,
            elinewidth=2,
            label='Data'
        )

        # Set labels
        plt.xlabel('Time [Weeks]', fontsize=18)
        plt.ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title('PD Simulation in Plasma of Healthy Volunteer', fontsize=22, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(title=f'{dose}', title_fontsize=18, fontsize=16, loc='lower right')
        plt.xlim(-200.0 / hours_per_week, 7000.0 / hours_per_week)
        plt.ylim(-118, 39)

        # Convert xlim from hours to weeks
        # if experiment == 'IVdose_20_HV':
        #     plt.xlim(-200.0 / hours_per_week, 7000.0 / hours_per_week)
        # elif experiment == 'IVdose_3_HV':
        #     plt.xlim(-140.0 / hours_per_week, 4400.0 / hours_per_week)
        # else:
        #     plt.xlim(-90.0 / hours_per_week, 3100.0 / hours_per_week)

        # plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # if experiment == 'IVdose_03_HV':
        #     plt.ylim(-118, 39)
        # else:
        #     plt.ylim(-118, 19)

        plt.tight_layout()

        # Save the figure
        save_path_svg = os.path.join(save_dir, f"{experiment}_PD_plot.svg")
        plt.savefig(save_path_svg, format='svg')
        plt.close()

def plot_two_doses_with_uncertainty(selected_params, acceptable_params, sims, PD_data, time_vectors, save_dir='../../../Results/HV/PD'):
    os.makedirs(save_dir, exist_ok=True)

    # Define colors and markers for each dose
    colors = ['#01947b', '#76b56e']
    markers = ['s', 'P']
    doses = ['0.3 mg/kg IV Dose', '20 mg/kg IV Dose']
    experiments = ['IVdose_03_HV', 'IVdose_20_HV']

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (experiment, color, dose) in enumerate(zip(experiments, colors, doses)):
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params:
            HV_params = np.delete(params.copy(), [11,16])
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=HV_params, reset=True)
                y_sim = sims[experiment].feature_data[:, 1]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Convert to weeks for plotting (1 week = 168 hours)
        hours_per_week = 168.0
        timepoints_weeks = timepoints / hours_per_week
        
        # Plot uncertainty range
        ax.fill_between(timepoints_weeks, y_min, y_max, color=color, alpha=0.3, label=f'{dose} Uncertainty')

        # Plot selected parameter set
        sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        y_selected = sims[experiment].feature_data[:, 1]
        ax.plot(timepoints_weeks, y_selected, color=color, label=f'{dose} Simulation', linewidth=3)

        # Plot experimental data
        marker = markers[i]
        exp_times_weeks = np.array(PD_data[experiment]['time']) / hours_per_week
        ax.errorbar(
            exp_times_weeks,
            PD_data[experiment]['BDCA2_median'],
            yerr=PD_data[experiment]['SEM'],
            marker=marker,
            markersize=8,
            color=color,
            linestyle='None',
            capsize=4,
            elinewidth=2,
            label='Data')

        ax.set_xlabel('Time [Weeks]', fontsize=18)
        ax.set_ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Intravenous Doses of 0.3 mg/kg and 20 mg/kg', fontsize=18)
        plt.suptitle('PD Simulation in Plasma of Healthy Volunteer', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.legend(title="Doses", title_fontsize=18, fontsize=16, loc='lower right')
        ax.set_xlim(-200.0 / hours_per_week, 7000.0 / hours_per_week)
        ax.set_ylim(-118, 39)
        plt.tight_layout()

        column_labels = ["0.3 mg/kg", "20 mg/kg"]

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
            ncol=2, 
            loc='upper center', 
            bbox_to_anchor=(0.9, 0.23),
            labelspacing=1.4,
            columnspacing=4.5,
            handletextpad=0.0,
            title="IV 0.3   IV 20", 
            title_fontsize=16,
            frameon=False
        )
        legend._legend_box.align = "center"

        row_labels = ["Uncertainty", "Simulation", "Data"]
        for i, label in enumerate(row_labels):
            ax.text(0.8, 0.14 - (i * 0.05), label, # Adjusted offsets
                    transform=ax.transAxes, ha='right', fontsize=16, va='center')
        
    save_path_svg = os.path.join(save_dir, f"{experiments[0]}_vs_{experiments[1]}_PD_plot.svg")
    plt.savefig(save_path_svg, format='svg')
    plt.close()


def plot_HV_vs_SLE_PD_simulation(final_params, final_params_SLE, acceptable_params, sims_HV, sims_SLE, PD_data, SLE_PD_data, time_vectors, save_dir='../../../Results/SLE/PD'):
    os.makedirs(save_dir, exist_ok=True)

    colors = plt.cm.Reds(np.linspace(0.7, 0.9, 2))
    labels = ["0.05 mg/kg IV Dose", "0.3 mg/kg IV Dose", "1 mg/kg IV Dose", "3 mg/kg IV Dose", "20 mg/kg IV Dose", "50 mg SC Dose"]

    for i, (experiment, label) in enumerate(zip(PD_data.keys(), labels)):
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
                y_sim_HV = sims_HV[experiment].feature_data[:, 1]
                y_min_HV = np.minimum(y_min_HV, y_sim_HV)
                y_max_HV = np.maximum(y_max_HV, y_sim_HV)

                sims_SLE[experiment].simulate(time_vector=timepoints, parameter_values=SLE_params, reset=True)
                y_sim_SLE = sims_SLE[experiment].feature_data[:, 1]
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

        ax.plot(time_weeks, sims_HV[experiment].feature_data[:, 1], color=colors[0], linewidth=2)
        ax.plot(time_weeks, sims_SLE[experiment].feature_data[:, 1], color=colors[1], linewidth=2, linestyle='dashed')

        exp_times_weeks = np.array(PD_data[experiment]['time']) / hours_per_week
        ax.errorbar(
            exp_times_weeks,
            PD_data[experiment]['BDCA2_median'],
            yerr=PD_data[experiment]['SEM'],
            fmt='o',
            markersize=6,
            color=colors[0],
            linestyle='None',
            capsize=3,
            label='HV Data'
        )

        if experiment == 'IVdose_20_HV':
            ax.errorbar(
                np.array(SLE_PD_data['IVdose_20_SLE']['time']) / hours_per_week,
                SLE_PD_data['IVdose_20_SLE']['BDCA2_median'],
                yerr=SLE_PD_data['IVdose_20_SLE']['SEM'],
                fmt='o',
                markersize=6,
                color=colors[1],
                linestyle='None',
                capsize=3,
                label='SLE Data'
            )

        ax.set_xlabel('Time [Weeks]', fontsize=18)
        ax.set_ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=18)
        ax.set_title("Healthy Volunteer vs SLE Patient", fontsize=18)
        plt.suptitle(f'PD Simulations in Plasma for a {label}', fontsize=22, fontweight='bold', x=0.54)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(-200.0 / hours_per_week, 7000.0 / hours_per_week)
        ax.set_ylim(-118, 39)
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
            bbox_to_anchor=(0.9, 0.24),
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
            ax.text(0.8, 0.14 - (i * 0.05), label, # Adjusted offsets
                    transform=ax.transAxes, ha='right', fontsize=16, va='center')

        # Save the figure
        save_path_svg = os.path.join(save_dir, f"PD_HV_vs_SLE_{experiment}.svg")
        plt.savefig(save_path_svg, format='svg')
        plt.close()


## Setup of the model
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
IV_005_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output('piecewise_constant', "IV_in",  t = PD_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output('piecewise_constant', "SC_in",  t = PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PD_data['SCdose_50_HV']['input']['SC_in']['f'])

# Creating simulation objects for each dose
model_sims_HV = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

model_sims_SLE = {
    'IVdose_005_HV': sund.Simulation(models = model_SLE, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model_SLE, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model_SLE, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model_SLE, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model_SLE, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model_SLE, activities = SC_50_HV, time_unit = 'h')
}

# Define time vectors for each dose
time_vectors = {exp: np.arange(-400, PD_data[exp]["time"][-1] + 5000, 1) for exp in PD_data}

# Plot all doses with uncertainty
# plot_all_doses_with_uncertainty(final_params, acceptable_params, model_sims_HV, PD_data, time_vectors)

# plot_two_doses_with_uncertainty(final_params, acceptable_params, model_sims_HV, PD_data, time_vectors)

plot_HV_vs_SLE_PD_simulation(final_params, final_params_SLE, acceptable_params, model_sims_HV, model_sims_SLE, PD_data, SLE_PD_data, time_vectors)
