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
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Load CLE Validation PK data
with open("../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load acceptable parameters for SLE
with open("../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params = json.load(f)

# Load final parameters for SLE
with open("../../Models/final_parameters_SLE.json", "r") as f:
    params = json.load(f)

# Install and load the model
sund.install_model('../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())
model = sund.load_model("mPBPK_SLE_model")

# Create activity objects for each dose
SC_50_CLE = sund.Activity(time_unit='h')
SC_50_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_CLE']['input']['SC_in']['t'],  f = PK_data['SCdose_50_CLE']['input']['SC_in']['f'])

SC_150_CLE = sund.Activity(time_unit='h')
SC_150_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_150_CLE']['input']['SC_in']['t'],  f = PK_data['SCdose_150_CLE']['input']['SC_in']['f'])

SC_450_CLE = sund.Activity(time_unit='h')
SC_450_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_450_CLE']['input']['SC_in']['t'],  f = PK_data['SCdose_450_CLE']['input']['SC_in']['f'])

# Create simulation objects for each dose
model_sims = {
    'SCdose_50_CLE': sund.Simulation(models=model, activities=SC_50_CLE, time_unit='h'),
    'SCdose_150_CLE': sund.Simulation(models=model, activities=SC_150_CLE, time_unit='h'),
    'SCdose_450_CLE': sund.Simulation(models=model, activities=SC_450_CLE, time_unit='h'),
}

# Create time vectors for each experiment
# The time vectors are created based on the maximum time in the PK_data for each experiment
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

def plot_model_uncertainty_with_validation_data(params, acceptable_params, sims, PK_data, time_vectors, save_dir='../../Results/Validation'):
    os.makedirs(save_dir, exist_ok=True)

    # Colors and markers for the plots
    colors = ['#6d65bf', '#6c5ce7', '#8c7ae6']
    markers = ['X', 'X', 'X']

    # Loop through each experiment and plot the uncertainty range and best parameter set
    for i, (experiment, color) in enumerate(zip(PK_data.keys(), colors)):
        plt.figure()
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

        # Plot best parameter set
        sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
        y = sims[experiment].feature_data[:, 0]
        plt.plot(timepoints, y, color=color, linewidth=2)

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

        # Labels and title
        plt.xlabel('Time [Hours]')
        plt.ylabel('Free Litifilimab Plasma Concentration (µg/ml)')
        plt.title(experiment)
        plt.tight_layout()

        # Save the plot
        save_path = os.path.join(save_dir, f"PK_validation_with_{experiment}.png")
        plt.savefig(save_path, format='png', dpi=600)
        plt.close()

# Plot the model uncertainty with validation data
plot_model_uncertainty_with_validation_data(params, acceptable_params, model_sims, PK_data, time_vectors)

