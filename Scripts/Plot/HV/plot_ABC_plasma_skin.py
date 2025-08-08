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

# Load acceptable parameters for mPBPK_model
with open("../../../Results/Acceptable params/acceptable_params.json", "r") as f:
    acceptable_params = json.load(f)

# Load final parameters for mPBPK_model
with open("../../../Models/final_parameters.json", "r") as f:
    params = json.load(f)

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

# Define a function to plot the ratio of anitbody concentration in skin and plasma for all doses
# Compare this value with the general ABC ratio of 0.157 by Shah and Betts (2012)
def plot_ABC_plasma_vs_skin(selected_params, sims, PK_data, time_vectors, save_dir='../../../Results/HV/PK'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    # Define colors and labels for each dose
    colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
    dose_labels = ['0.05 mg/kg', '0.3 mg/kg', '1 mg/kg', '3 mg/kg', '10 mg/kg', '20 mg/kg', '50 mg/kg SC']

    # Create empty lists to store all skin and plasma concentrations
    all_skin = []
    all_plasma = []

    # Iterate through each experiment and simulate PK in plasma and skin
    for i, (experiment, color, label) in enumerate(zip(PK_data.keys(), colors, dose_labels)):
        timepoints = time_vectors[experiment]
        sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
        skin = sims[experiment].feature_data[:, 2]
        plasma = sims[experiment].feature_data[:, 0]

        # Only keep points where both values are positive
        valid = (skin > 0) & (plasma > 0)
        skin = skin[valid]
        plasma = plasma[valid]

        # Plot the skin vs plasma concentration
        plt.scatter(plasma, skin, color=color, label=label, alpha=0.7)
        all_skin.extend(skin)
        all_plasma.extend(plasma)

    # Calculate the minimum and maximum values for skin and plasma concentrations
    x_min = min(all_plasma)
    x_max = max(all_plasma)
    x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)

    # Plot the reference lines for the general ABC ratio
    plt.plot(x_fit, 0.157 * x_fit, 'k-')
    plt.plot(x_fit, 0.0785 * x_fit, 'k--')
    plt.plot(x_fit, 0.314 * x_fit, 'k--')

    # Set scale, labels, title, and legend
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Plasma Concentration (µg/ml)')
    plt.ylabel('Skin Concentration (µg/ml)')
    plt.title('Model Simulations for Skin vs Plasma Concentration in Healthy Volunteers')
    plt.legend()

    # Save the plot
    save_path = os.path.join(save_dir, "Skin vs Plasma Concentration.png")
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()

# Run the function to plot the skin vs plasma concentration for all doses
plot_ABC_plasma_vs_skin(params, model_sims, PK_data, time_vectors)

