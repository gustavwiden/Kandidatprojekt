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

# Load model and data

with open("../../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()
    
sund.install_model('../../../Models/mPBPK_model.txt')
model = sund.load_model("mPBPK_model")

with open("../../../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

with open('../../../Results/Acceptable params/acceptable_params.json', 'r') as f:
    acceptable_params = json.load(f)

# Bodyweight for subject in kg
bodyweight = 73

# Create activity objects for each dose
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h')
IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_20_HV']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PD_data['SCdose_50_HV']['input']['SC_in']['f'])

# Create simulation objects for each dose
model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}
# Time vectors for each experiment
time_vectors = {exp: np.arange(-50, PD_data[exp]["time"][-1] + 0.01, 1) for exp in PD_data}

HV_params = [0.81995, 0.009023581987003631, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6927716105886019, 0.95, 0.7960584853135797, 0.2, 0.007911517932177177, 2.22, 1.14185149185025, 14000.0]

# Define a function to plot one PD dataset
def plot_PD_dataset(PD_data, face_color='k'):
    plt.errorbar(PD_data['time'], PD_data['BDCA2_median'], PD_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BDCA2 levels on pDCs, percentage change from baseline')

# Define a function to plot all PD datasets
def plot_PD_data(PD_data, face_color='k'):
    for experiment in PD_data:
        plt.figure()
        plot_PD_dataset(PD_data[experiment], face_color=face_color)
        plt.title(experiment)

# Define a function to plot the simulation results
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PD_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Plot uncertainty and data for each dose in a separate figure
def plot_sim_with_uncertainty(params, acceptable_params, sims, PD_data, time_vectors, save_dir='../Results/HV_results/PD', feature_to_plot='PD_sim'):
    os.makedirs(save_dir, exist_ok=True)
    color_uncertainty = '#1b7837'
    color_sim = 'blue'

    for experiment in PD_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        y_min = np.full_like(timepoints, np.inf)
        y_max = np.full_like(timepoints, -np.inf)

        # Calculate uncertainty range
        for p in acceptable_params:
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=p, reset=True)
                y_sim = sims[experiment].feature_data[:, sims[experiment].feature_names.index(feature_to_plot)]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except Exception:
                continue

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color=color_uncertainty, alpha=0.3, label='Uncertainty')

        # Plot data points
        plot_PD_dataset(PD_data[experiment])

        # Plot simulation with HV_params
        plot_sim(params, sims[experiment], timepoints, color=color_sim, feature_to_plot=feature_to_plot)

        plt.title(experiment)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"PD_{experiment}_uncertainty.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

# Run the plotting function
plot_sim_with_uncertainty(HV_params, acceptable_params, model_sims, PD_data, time_vectors)

