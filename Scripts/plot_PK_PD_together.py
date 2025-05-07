import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Functions')))

from functions import plot_PK_PD_sim_with_data
import json
import numpy as np
import sund

# Ensure the Results folder exists
results_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Results'))
os.makedirs(results_folder, exist_ok=True)

# Load the PK and PD data
with open("../Data/PK_data.json", "r") as pk_file:
    PK_data = json.load(pk_file)

with open("../Data/PD_data.json", "r") as pd_file:
    PD_data = json.load(pd_file)

# Load the model
sund.install_model('../Models/mPBPK_model.txt')
first_model = sund.load_model("mPBPK_model")

# Create activities for the different doses
bodyweight = 70  # Bodyweight in kg
first_model_sims = {}

for experiment in PK_data:
    activity = sund.Activity(time_unit='h')
    input_data = PK_data[experiment]['input']['IV_in'] if 'IV_in' in PK_data[experiment]['input'] else PK_data[experiment]['input']['SC_in']
    activity.add_output(
        sund.PIECEWISE_CONSTANT,
        "IV_in" if 'IV_in' in PK_data[experiment]['input'] else "SC_in",
        t=input_data['t'],
        f=bodyweight * np.array(input_data['f']) if 'IV_in' in PK_data[experiment]['input'] else np.array(input_data['f'])
    )
    first_model_sims[experiment] = sund.Simulation(models=first_model, activities=activity, time_unit='h')

# Generate time vectors for each experiment
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 2000, 1) for exp in PK_data}

# Define parameters (example parameters, replace with actual values)
params = [0.679, 0.01, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.769, 0.95, 0.605, 0.2, 5.896, 
          13.9, 0.421, 1.92e-4, 5e-8, 8, 8, 0.525]

# Plot and save each experiment
for experiment in PK_data:
    fig = plot_PK_PD_sim_with_data(params, first_model_sims, experiment, PK_data, PD_data, time_vectors)
    if fig:  # Only save if a figure was returned
        plot_path = os.path.join(results_folder, f"{experiment}_PK_PD_simulation.png")
        fig.savefig(plot_path)
        print(f"Saved plot for {experiment} to {plot_path}")