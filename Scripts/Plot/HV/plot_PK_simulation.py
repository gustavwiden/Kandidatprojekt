# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
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
acceptable_params_PL = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()

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

    # Define labels and positions for each dose
    dose_labels = {
        'IVdose_005_HV': '0.05 IV',
        'IVdose_03_HV':  '0.3 IV',
        'IVdose_1_HV':   '1 IV',
        'IVdose_3_HV':   '3 IV',
        'IVdose_10_HV':  '10 IV',
        'IVdose_20_HV':  '20 IV',
        'SCdose_50_HV':  '50 SC'
    }

    label_positions = {
        'IVdose_005_HV': (400, 0.06),
        'IVdose_03_HV':  (300, 0.6),
        'IVdose_1_HV':   (1480, 0.11),
        'IVdose_3_HV':   (780, 7),
        'IVdose_10_HV':  (780, 22),
        'IVdose_20_HV':  (1900, 66),
        'SCdose_50_HV':  (800, 0.16),
    }
    
    # Loop through each experiment
    for i, (experiment, color) in enumerate(zip(PK_data.keys(), colors)):
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params:
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                y_sim = sims[experiment].feature_data[:, 0]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)

        # Plot selected parameter set
        sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        y_selected = sims[experiment].feature_data[:, 0]
        plt.plot(timepoints, y_selected, color=color, linewidth=2)

        # Plot experimental data
        marker = markers[i]
        plt.errorbar(
            PK_data[experiment]['time'],
            PK_data[experiment]['BIIB059_mean'],
            yerr=PK_data[experiment]['SEM'],
            fmt=marker,
            markersize=6,
            color=color,
            linestyle='None',
            capsize=3
        )

        # Add manually placed labels
        if experiment in label_positions:
            label_x, label_y = label_positions[experiment]
            plt.text(label_x, label_y, dose_labels.get(experiment, experiment),
                     color=color, fontsize=18, weight='bold')

    # Set labels
    plt.xlabel('Time [Hours]', fontsize=20)
    plt.ylabel('Free Litifilimab Plasma Concentration (µg/ml)', fontsize=20)
    plt.yscale('log')
    plt.ylim(0.005, 1000)
    plt.xlim(-25, 2750)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()

    # Text to describe the figure
    plt.annotate(
        'Simulation',
        xy=(1470, 60),  # Arrow's coordinates
        xytext=(1500, 200),  # Text coordinates
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=18
    )

    plt.annotate(
        'Uncertainty',
        xy=(2450, 0.8), # Arrow's coordinates
        xytext=(2050, 0.1),  # Text coordinates
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=18
    )

    plt.annotate(
        'Data',
        xy=(1025, 94),  # Arrow's coordinates
        xytext=(1250, 180),  # Text coordinates
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=18
    )

    # Save the figure
    save_path = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.png")
    plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)
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
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data['SCdose_50_HV']['input']['SC_in']['f'])

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
plot_all_doses_with_uncertainty(params, acceptable_params_PL, model_sims, PK_data, time_vectors)
