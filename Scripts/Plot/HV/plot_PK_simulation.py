# # Importing the necessary libraries
# import os
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import sund
# import json
# from scipy.stats import chi2
# from scipy.optimize import Bounds
# from scipy.optimize import differential_evolution
# import sys
# import csv
# import random
# import requests

# from json import JSONEncoder
# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)

# # Open the mPBPK_model.txt file and read its contents
# with open("../../../Models/mPBPK_model.txt", "r") as f:
#     lines = f.readlines()

# # Open the data file and read its contents
# with open("../../../Data/PK_data.json", "r") as f:
#     PK_data = json.load(f)

# # Define a function to plot one PK_dataset
# def plot_PK_dataset(PK_data, face_color='k'):
#     plt.errorbar(PK_data['time'], PK_data['BIIB059_mean'], PK_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
#     plt.xlabel('Time [Hours]')
#     plt.ylabel('BIIB059 serum conc. (µg/ml)')

# # Definition of a function that plots the simulation
# def plot_sim(params, sim, timepoints, color='g', feature_to_plot='PK_sim'):
#     sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
#     feature_idx = sim.feature_names.index(feature_to_plot)
#     plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# # Definition of the function that plots all PK simulations and saves them to Results folder
# def plot_sim_with_PK_data(params, sims, PK_data, color='g', save_dir='../Results'):
#     os.makedirs(save_dir, exist_ok=True)

#     for experiment in PK_data:
#         plt.figure()
#         timepoints = time_vectors[experiment]
#         plot_sim(params, sims[experiment], timepoints, color)
#         plot_PK_dataset(PK_data[experiment])
#         plt.title(experiment)

#         # Save figure with PK-specific name
#         filename = f"PK_{experiment}_simulation.png"
#         save_path = os.path.join(save_dir, filename)
#         plt.savefig(save_path, bbox_inches='tight')
#         filename = f"PK_{experiment}_simulation.pdf"
#         plt.close()

# # Save figure with all doses together
# def plot_all_doses_together(params, sims, PK_data, time_vectors, save_dir='../Results', feature_to_plot='PK_sim'):
#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure(figsize=(12, 7))

#     # # Change background color for poster
#     # plt.gcf().patch.set_facecolor('#fcf5ed')
#     # plt.gca().set_facecolor('#fcf5ed')

#     colors = ['#1b7837', '#01947b', '#628759', '#70b5aa','#35978f', '#76b56e', '#6d65bf']
#     markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

#     dose_labels = {
#         'IVdose_005_HV': '0.05 IV',
#         'IVdose_03_HV':  '0.3 IV',
#         'IVdose_1_HV':   '1 IV',
#         'IVdose_3_HV':   '3 IV',
#         'IVdose_10_HV':  '10 IV',
#         'IVdose_20_HV':  '20 IV',
#         'SCdose_50_HV':  '50 SC'
#     }

#     label_positions = {
#         'IVdose_005_HV': (400, 0.05),
#         'IVdose_03_HV':  (300, 0.8),
#         'IVdose_1_HV':   (1550, 1.25),
#         'IVdose_3_HV':   (1760, 5),
#         'IVdose_10_HV':  (1750, 20),
#         'IVdose_20_HV':  (1600, 55),
#         'SCdose_50_HV':  (1080, 0.75),
#     }

#     for i, (experiment, color) in enumerate(zip(PK_data.keys(), colors)):
#         timepoints = time_vectors[experiment]
#         sim = sims[experiment]
#         sim.simulate(time_vector=timepoints, parameter_values=params, reset=True)
#         feature_idx = sim.feature_names.index(feature_to_plot)

#         # Plot simulation data
#         plt.plot(sim.time_vector, sim.feature_data[:, feature_idx],
#                  color=color, linewidth=2, label=None)

#         # Plot labels
#         if experiment in label_positions:
#             label_x, label_y = label_positions[experiment]
#             plt.text(label_x, label_y, dose_labels.get(experiment, experiment),
#                      color=color, fontsize=18, weight='bold')

#         # Data points
#         marker = markers[i]
#         plt.errorbar(
#             PK_data[experiment]['time'],
#             PK_data[experiment]['BIIB059_mean'],
#             yerr=PK_data[experiment]['SEM'],
#             fmt=marker,
#             markersize=6,
#             color=color,
#             label=dose_labels.get(experiment, experiment),
#             linestyle='None',
#             capsize=3
#         )

#     plt.xlabel('Time [Hours]', fontsize=18)
#     plt.ylabel('BIIB059 plasma conc. (µg/ml)', fontsize=18)
#     plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
#     plt.text(10, 2, 'Baseline', color='gray', fontsize=18)

#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.25)

#     plt.yscale('log')
#     plt.ylim(0.03, 700)
#     plt.xlim(-25, 2750)

#     save_path_svg = os.path.join(save_dir, "PK_all_doses_together.svg")
#     plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
#     save_path_pdf = os.path.join(save_dir, "PK_all_doses_together.pdf")
#     plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)

#     print(f"Saving SVG to: {save_path_svg}")
#     print(f"Saving PDF to: {save_path_pdf}")

#     plt.close()

# def plot_all_doses_with_uncertainty(selected_params, acceptable_params, sims, PK_data, time_vectors, save_dir='../Results', feature_to_plot='PK_sim'):
#     os.makedirs(save_dir, exist_ok=True)
#     plt.figure(figsize=(12, 7))

#     # # Change background color for poster
#     # plt.gcf().patch.set_facecolor('#fcf5ed')
#     # plt.gca().set_facecolor('#fcf5ed')

#     colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
#     markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

#     dose_labels = {
#         'IVdose_005_HV': '0.05 IV',
#         'IVdose_03_HV':  '0.3 IV',
#         'IVdose_1_HV':   '1 IV',
#         'IVdose_3_HV':   '3 IV',
#         'IVdose_10_HV':  '10 IV',
#         'IVdose_20_HV':  '20 IV',
#         'SCdose_50_HV':  '50 SC'
#     }

#     label_positions = {
#         'IVdose_005_HV': (400, 0.07),
#         'IVdose_03_HV':  (300, 0.4),
#         'IVdose_1_HV':   (1480, 0.10),
#         'IVdose_3_HV':   (1570, 2),
#         'IVdose_10_HV':  (770, 21),
#         'IVdose_20_HV':  (1900, 80),
#         'SCdose_50_HV':  (900, 0.17),
#     }
    

#     for i, (experiment, color) in enumerate(zip(PK_data.keys(), colors)):
#         timepoints = time_vectors[experiment]
#         y_min = np.full_like(timepoints, np.inf)
#         y_max = np.full_like(timepoints, -np.inf)

#         # Calculate uncertainty range
#         for params in acceptable_params:
#             try:
#                 sims[experiment].simulate(time_vector=timepoints, parameter_values=params, reset=True)
#                 y_sim = sims[experiment].feature_data[:, 0]
#                 y_min = np.minimum(y_min, y_sim)
#                 y_max = np.maximum(y_max, y_sim)
#             except RuntimeError as e:
#                 if "CV_ERR_FAILURE" in str(e):
#                     print(f"Skipping unstable parameter set for {experiment}")
#                 else:
#                     raise e

#         # Plot uncertainty range
#         plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3)

#         # Plot selected parameter set
#         sims[experiment].simulate(time_vector=timepoints, parameter_values=selected_params, reset=True)
#         y_selected = sims[experiment].feature_data[:, 0]
#         plt.plot(timepoints, y_selected, color=color, linewidth=2)

#         # Plot experimental data
#         marker = markers[i]
#         plt.errorbar(
#             PK_data[experiment]['time'],
#             PK_data[experiment]['BIIB059_mean'],
#             yerr=PK_data[experiment]['SEM'],
#             fmt=marker,
#             markersize=6,
#             color=color,
#             linestyle='None',
#             capsize=3
#         )

#         # Add manually placed labels
#         if experiment in label_positions:
#             label_x, label_y = label_positions[experiment]
#             plt.text(label_x, label_y, dose_labels.get(experiment, experiment),
#                      color=color, fontsize=18, weight='bold')

#     plt.xlabel('Time [Hours]', fontsize=22)
#     plt.ylabel('BIIB059 Plasma Concentration (µg/ml)', fontsize=22)
#     plt.yscale('log')
#     plt.ylim(0.03, 700)
#     plt.xlim(-25, 2750)
#     plt.tick_params(axis='both', which='major', labelsize=22)
#     plt.tight_layout()

#     # Text to describe the figure
#     plt.annotate(
#         'Simulation',
#         xy=(1470, 60),  # Arrow's coordinates (adjust as needed)
#         xytext=(1500, 200),  # Text coordinates
#         arrowprops=dict(facecolor='black', arrowstyle='->'),
#         fontsize=18
#     )

#     plt.annotate(
#         'Uncertainty',
#         xy=(2500, 0.8), # Arrow's coordinates
#         xytext=(2100, 0.1),  # Text coordinates
#         arrowprops=dict(facecolor='black', arrowstyle='->'),
#         fontsize=18
#     )

#     plt.annotate(
#         'Data',
#         xy=(1025, 94),  # Arrow's coordinates
#         xytext=(1250, 180),  # Text coordinates
#         arrowprops=dict(facecolor='black', arrowstyle='->'),
#         fontsize=18
#     )

#     #Save and show figure
#     save_path_pdf = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.pdf")
#     plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', dpi=300)

#     save_path_svg = os.path.join(save_dir, "PK_all_doses_together_with_uncertainty.svg")
#     plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
#     plt.show()

#     plt.close()


# ## Setup of the model
# # Install the model
# sund.install_model('../../../Models/mPBPK_model.txt')
# print(sund.installed_models())

# # Load the model object
# first_model = sund.load_model("mPBPK_model")

# # Creating activities for the different doses
# bodyweight = 70 # Bodyweight for subject in kg

# IV_005_HV = sund.Activity(time_unit='h')
# IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_005_HV']['input']['IV_in']['f']))

# IV_03_HV = sund.Activity(time_unit='h')
# IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_03_HV']['input']['IV_in']['f']))

# IV_1_HV = sund.Activity(time_unit='h')
# IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_1_HV']['input']['IV_in']['f']))

# IV_3_HV = sund.Activity(time_unit='h')
# IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_3_HV']['input']['IV_in']['f']))

# IV_10_HV = sund.Activity(time_unit='h')
# IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_10_HV']['input']['IV_in']['f']))

# IV_20_HV = sund.Activity(time_unit='h')
# IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data['IVdose_20_HV']['input']['IV_in']['f']))

# SC_50_HV = sund.Activity(time_unit='h')
# SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data['SCdose_50_HV']['input']['SC_in']['f'])

# first_model_sims = {
#     'IVdose_005_HV': sund.Simulation(models = first_model, activities = IV_005_HV, time_unit = 'h'),
#     'IVdose_03_HV': sund.Simulation(models = first_model, activities = IV_03_HV, time_unit = 'h'),
#     'IVdose_1_HV': sund.Simulation(models = first_model, activities = IV_1_HV, time_unit = 'h'),
#     'IVdose_3_HV': sund.Simulation(models = first_model, activities = IV_3_HV, time_unit = 'h'),
#     'IVdose_10_HV': sund.Simulation(models = first_model, activities = IV_10_HV, time_unit = 'h'),
#     'IVdose_20_HV': sund.Simulation(models = first_model, activities = IV_20_HV, time_unit = 'h'),
#     'SCdose_50_HV': sund.Simulation(models = first_model, activities = SC_50_HV, time_unit = 'h')
# }

# time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

# def fcost(params, sims, PK_data):
#     cost = 0
#     for dose in PK_data:
#         try:
#             sims[dose].simulate(time_vector = PK_data[dose]["time"], parameter_values = params, reset = True)
#             PK_sim = sims[dose].feature_data[:,0]
#             y = PK_data[dose]["BIIB059_mean"]
#             SEM = PK_data[dose]["SEM"] 
#             cost += np.sum(np.square(((PK_sim - y) / SEM)))
#         except Exception as e:
#             if "CVODE" not in str(e):
#                 print(str(e))
#             return 1e30
#     return cost

# params_HV = [0.81995, 0.009023581987003631, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6927716105886019, 0.95, 0.7960584853135797, 0.2, 0.007911517932177177, 2.22, 1.14185149185025, 14000.0] # Optimized parameters

# PK_cost_HV = fcost(params_HV, first_model_sims, PK_data)
# print(f"Cost of the PK HV model: {PK_cost_HV}")

# dgf = sum(np.count_nonzero(np.isfinite(PK_data[exp]["SEM"])) for exp in PK_data)
# chi2_limit = chi2.ppf(0.95, dgf)
# print(f"Chi2 limit: {chi2_limit}")
# print(f"Cost > limit (rejected?): {PK_cost_HV > chi2_limit}")

# # Plotting the simulation with PK data for each dose and all doses together
# #plot_sim_with_PK_data(params_HV, first_model_sims, PK_data)
# #plot_all_doses_together(params_HV, first_model_sims, PK_data, time_vectors)

# # Load acceptable parameters
# with open('../../../Results/Acceptable params/acceptable_params.json', 'r') as f:
#     acceptable_params = json.load(f)

# # Plot all doses with uncertainty
# plot_all_doses_with_uncertainty(params_HV, acceptable_params, first_model_sims, PK_data, time_vectors)

# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import sund
import json
from scipy.stats import chi2
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
import sys
import csv
import random
import requests

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

with open("../../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)

with open('../../../Results/Acceptable params/acceptable_params.json', 'r') as f:
    acceptable_params = json.load(f)

# Bodyweight for subject in kg
bodyweight = 70

# Create activity objects for each dose
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

# Create simulation objects for each dose
model_sims = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h')
}
# Time vectors for each experiment
time_vectors = {exp: np.arange(-50, PK_data[exp]["time"][-1] + 0.01, 1) for exp in PK_data}

HV_params = [0.81995, 0.009023581987003631, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6927716105886019, 0.95, 0.7960584853135797, 0.2, 0.007911517932177177, 2.22, 1.14185149185025, 14000.0]

# Define a function to plot one PK dataset
def plot_PK_dataset(PK_data, face_color='k'):
    plt.errorbar(PK_data['time'], PK_data['BIIB059_mean'], PK_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BIIB059 serum conc. (µg/ml)')

# Define a function to plot all PK datasets
def plot_PK_data(PK_data, face_color='k'):
    for experiment in PK_data:
        plt.figure()
        plot_PK_dataset(PK_data[experiment], face_color=face_color)
        plt.title(experiment)

# Define a function to plot the simulation results
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='PK_sim'):
    sim.simulate(time_vector = timepoints, parameter_values = params, reset = True)
    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)

# Plot uncertainty and data for each dose in a separate figure
def plot_sim_with_uncertainty(params, acceptable_params, sims, PK_data, time_vectors, save_dir='../Results/HV_results/PK', feature_to_plot='PK_sim'):
    os.makedirs(save_dir, exist_ok=True)
    color_uncertainty = '#1b7837'
    color_sim = 'blue'

    for experiment in PK_data:
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
        plot_PK_dataset(PK_data[experiment])

        # Plot simulation with HV_params
        plot_sim(params, sims[experiment], timepoints, color=color_sim, feature_to_plot=feature_to_plot)

        plt.title(experiment)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"PK_{experiment}_uncertainty.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

# Run the plotting function
plot_sim_with_uncertainty(HV_params, acceptable_params, model_sims, PK_data, time_vectors)


