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

# Open the data file and read its contents
with open("../../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params.json", "r") as f:
    acceptable_params = json.load(f)

def plot_all_doses_with_uncertainty(selected_params, acceptable_params, sims, PK_data, time_vectors, save_dir='../../../Results/HV/PK', feature_to_plot='PK_sim'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 7))

    # # Change background color for poster
    # plt.gcf().patch.set_facecolor('#fcf5ed')
    # plt.gca().set_facecolor('#fcf5ed')

    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

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
        'IVdose_005_HV': (400, 0.1),
        'IVdose_03_HV':  (200, 0.6),
        'IVdose_1_HV':   (1480, 0.11),
        'IVdose_3_HV':   (1570, 2),
        'IVdose_10_HV':  (770, 19),
        'IVdose_20_HV':  (1900, 66),
        'SCdose_50_HV':  (820, 0.2),
    }
    

    for i, (experiment, color) in enumerate(zip(PK_data.keys(), colors)):
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, np.inf)
        y_max = np.full_like(timepoints, -np.inf)

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

    plt.xlabel('Time [Hours]', fontsize=22)
    plt.ylabel('BIIB059 Plasma Concentration (µg/ml)', fontsize=22)
    plt.yscale('log')
    plt.ylim(0.01, 1000)
    plt.xlim(-25, 2750)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.tight_layout()

    # Text to describe the figure
    plt.annotate(
        'Simulation',
        xy=(1470, 60),  # Arrow's coordinates (adjust as needed)
        xytext=(1500, 200),  # Text coordinates
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=18
    )

    plt.annotate(
        'Uncertainty',
        xy=(2500, 0.8), # Arrow's coordinates
        xytext=(2100, 0.1),  # Text coordinates
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

    save_path_png = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.svg")
    plt.savefig(save_path_png, format='svg', bbox_inches='tight')
    plt.show()

    plt.close()

## Setup of the model
# Install the model
sund.install_model('../../../Models/mPBPK_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_model")

# Creating activities for the different doses
bodyweight = 73 # Bodyweight for subject in kg

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

model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}

time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

params_HV =[0.6275806018256461, 0.012521665343092613, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.63, 0.95, 0.7965420036627042, 0.2, 0.005402247272171798, 67.02886707793152, 5.539999999999999, 2497.3959216508456]

# Plot all doses with uncertainty
plot_all_doses_with_uncertainty(params_HV, acceptable_params, model_sims, PK_data, time_vectors)
