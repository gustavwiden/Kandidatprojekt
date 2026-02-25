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

# Load PK data
with open("../../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load acceptable parameters for mPBPK_model from PL
acceptable_params = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()

# Load final parameters for mPBPK_model from optimization
with open("../../../Models/final_parameters.json", "r") as f:
    params = json.load(f)

# Define a function to plot all doses with uncertainty in the same figure
def plot_all_doses_with_uncertainty(selected_params, acceptable_params, sims, PK_data, time_vectors, save_dir='../../../Results/HV/PK'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

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
        plt.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.3, label="Uncertainty")

        # Plot selected parameter set (simulate using hours, plot using weeks)
        sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        y_selected = sims[experiment].feature_data[:, 0]
        plt.plot(time_weeks, y_selected, color=color, linewidth=2, label="Simulation")

        # Plot experimental data (convert times to weeks)
        marker = markers[i]
        exp_times_weeks = np.array(PK_data[experiment]['time']) / 168.0
        plt.errorbar(
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

    plt.xlabel('Time [Weeks]', fontsize=18)
    plt.ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
    plt.title('PK Simulations in Plasma of Healthy Volunteers', fontsize=22, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yscale('log')
    plt.ylim(0.005, 1000)
    # convert previous xlim from hours to weeks
    plt.xlim(-25.0 / hours_per_week, 2750.0 / hours_per_week)
    plt.tick_params(axis='both', which='major', labelsize=16)   
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) 

    column_labels = ["0.05 mg/kg", "0.3 mg/kg", "1 mg/kg", "3 mg/kg", "10 mg/kg", "20 mg/kg", "SC 50 mg"]

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
        bbox_to_anchor=(0.54, -0.11),
        labelspacing=0.9,
        columnspacing=7.2,
        handletextpad=0.0,
        title="0.05 mg/kg  0.3 mg/kg  1.0 mg/kg  3.0 mg/kg  10 mg/kg  20 mg/kg  SC 50 mg", 
        title_fontsize=16,
        frameon=False
    )
    legend._legend_box.align = "center"

    row_labels = ["IV", "Uncertainty", "Simulation", "Data"]
    for i, label in enumerate(row_labels):
        ax.text(0.13, -0.152 - (i * 0.05), label, # Adjusted offsets
                transform=ax.transAxes, ha='right', fontsize=16, va='center')

    # Save the figure
    # save_path = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.png")
    # plt.savefig(save_path, format='png', dpi=600)
    # plt.close()

    save_path_svg = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.svg")
    plt.savefig(save_path_svg, format='svg')
    plt.close()

# Install the model
sund.install_model('../../../Models/mPBPK_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_model")

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
model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

# Define time vectors for each dose
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

# Plot all doses with uncertainty
plot_all_doses_with_uncertainty(params, acceptable_params, model_sims, PK_data, time_vectors)
