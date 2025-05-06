
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
 
# Open the mPBPK_model.txt file and read its contents
with open("mPBPK_model.txt", "r") as f:
    lines = f.readlines()

# Open the data file and read its contents
with open("PD_data.json", "r") as f:
    PD_data = json.load(f)


# Define a function to plot one PD_dataset
def plot_PD_dataset(PD_data, face_color='k'):
    plt.errorbar(PD_data['time'], PD_data['BDCA2_median'], PD_data['SEM'], linestyle='None', marker='o', markerfacecolor=face_color, color='k')
    plt.xlabel('Time [Hours]')
    plt.ylabel('BDCA2 levels on pDCs, percentage change from baseline. (µg/ml)')


# Defininition of the function that plot all PD_datasets
def plot_PD_data(PD_data, face_color='k'):
    for experiment in PD_data:
        plt.figure()
        plot_PD_dataset(PD_data[experiment], face_color=face_color)
        plt.title(experiment)


## Setup of the model

# Install the model
sund.install_model('mPBPK_model.txt')
print(sund.installed_models())

# Load the model object
first_model = sund.load_model("mPBPK_model")



# Creating activities for the different doses
bodyweight = 70 # Bodyweight for subject in kg

IV_005_HV = sund.Activity(time_unit='h') # Intravenous dose of 0.05 mg/kg in Healthy volunteer (HV)
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h') # Intravenous dose of 0.3 mg/kg in HV
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h') # Intravenous dose of 1 mg/kg in HV
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h') # Intravenous dose of 3 mg/kg in HV
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_3_HV']['input']['IV_in']['f']))

# IV_10_HV = sund.Activity(time_unit='h') # Intravenous dose of 10 mg/kg in HV
# IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_HV = sund.Activity(time_unit='h') # Intravenous dose of 20 mg/kg in HV
IV_20_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_20_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_20_HV']['input']['IV_in']['f']))

# IV_20_SLE = sund.Activity(time_unit='h') # Intravenous dose of 20 mg/kg in SLE affected patients
# IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PD_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PD_data['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h') # Subcutaneous (SC) dose of 50 mg/kg in HV
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = PD_data['SCdose_50_HV']['input']['SC_in']['f'])


# Addition of the different simulation objects into a dictionary 
first_model_sims = {}
first_model_sims['IVdose_005_HV'] = sund.Simulation(models = first_model, activities = IV_005_HV, time_unit = 'h')
first_model_sims['IVdose_03_HV'] = sund.Simulation(models = first_model, activities = IV_03_HV, time_unit = 'h')
first_model_sims['IVdose_1_HV'] = sund.Simulation(models = first_model, activities = IV_1_HV, time_unit = 'h')
first_model_sims['IVdose_3_HV'] = sund.Simulation(models = first_model, activities = IV_3_HV, time_unit = 'h')
# first_model_sims['IVdose_10_HV'] = sund.Simulation(models = first_model, activities = IV_10_HV, time_unit = 'h')
first_model_sims['IVdose_20_HV'] = sund.Simulation(models = first_model, activities = IV_20_HV, time_unit = 'h')
# first_model_sims['IVdose_20_SLE'] = sund.Simulation(models = first_model, activities = IV_20_SLE, time_unit = 'h')
first_model_sims['SCdose_50_HV'] = sund.Simulation(models = first_model, activities = SC_50_HV, time_unit = 'h')

# Precompute time vectors for each dose in PD_data
time_vectors = {}
for experiment in PD_data:
    max_time = PD_data[experiment]["time"][-1]  # Get the maximum time from PD_data
    time_vectors[experiment] = np.arange(-10, max_time + 1000, 1)  # Time vector starting at -10 hours

# Defininition of a function that plot the simulation
def plot_sim(params, sim, timepoints, color='b', feature_to_plot='y_sim'):
    # Setup, simulate, and plot the model
    sim.simulate(time_vector = timepoints,
                parameter_values = params,
                    reset = True)

    feature_idx = sim.feature_names.index(feature_to_plot)
    plt.plot(sim.time_vector, sim.feature_data[:,feature_idx], color)


# Definition of a function that plot the simulations together with the PD_data
def plot_sim_with_PD_data(params, sims, PD_data, color='b'):
    for experiment in PD_data:
        plt.figure()
        timepoints = time_vectors[experiment]
        plot_sim(params, sims[experiment], timepoints, color)
        plot_PD_dataset(PD_data[experiment])
        plt.title(experiment)


## Plotting the first model with different parmeter values for either HV clearence or SLE clearence, remember to put a comment over the one not used 
# HV_CL


# Cost calculation 
def fcost(params, sims, PD_data):
    cost = 0
    
    for dose in PD_data.keys():
        try:
            sims[dose].simulate(time_vector = PD_data[dose]["time"], 
                            parameter_values = params,
                            reset = True)

            y_sim = sims[dose].feature_data[:,0]
            y = PD_data[dose]["BDCA2_median"]
            SEM = PD_data[dose]["SEM"] 

            cost += np.sum(np.square(((y_sim - y) / SEM)))

        except Exception as e:
            if "CVODE" not in str(e):
                print(str(e))
            return 1e30
    return cost


params_M1 = [0.713, 0.00975, 2600, 1810, 6300, 4370, 2600, 10.29, 29.58, 80.96, 0.63, 0.95, 0.75, 
0.20, 5.52, 10.7, 0.547, 1.92e-4, 5e-8, 4, 4, 0.35]

cost_M1 =  fcost(params_M1, first_model_sims, PD_data)
print(f"Cost of the M1 model: {cost_M1}")


dgf=0
for experiment in PD_data:
    dgf += np.count_nonzero(np.isfinite(PD_data[experiment]["SEM"]))
chi2_limit = chi2.ppf(0.95, dgf) # Here, 0.95 corresponds to 0.05 significance level
print(f"Chi2 limit: {chi2_limit}")
print(f"Cost > limit (rejected?): {cost_M1>chi2_limit}")

plot_sim_with_PD_data(params_M1, first_model_sims, PD_data)

plt.show()



# Improving the agreement using optimization methods 
def callback(x, file_name):
    with open(f"./{file_name}.json",'w') as file:
        out = {"x": x}
        json.dump(out,file, cls=NumpyArrayEncoder)

args_M1 = (first_model_sims, PD_data)
params_M1_log = np.log(params_M1)
 

# Define relative bounds for each parameter
# The relative bounds are defined as a factor of the parameter value. For example, if the parameter value is 1 and the bound factor is 1.2, the bounds will be [0.8333, 1.2].
# = [F, ka, kd, Vp, Vs, V1, V2, VL, Ls, L1, L2, RCS, RC1, RC2, RCL, HV_CL, Vm, Km, ksynp, ksyns, kintp, kints, kd]
bound_factors = [1.1, 1.1, 1, 1, 1, 1, 1, 1, 1, 1, 1.2, 1, 1.2, 1, 1.2, 1.2, 1.2, 2, 2, 2, 2, 1.5]

# Calculate lower and upper bounds for each parameter
lower_bounds = np.log(params_M1) - np.log(bound_factors)
upper_bounds = np.log(params_M1) + np.log(bound_factors)

# Create the bounds object
bounds_M1_log = Bounds(lower_bounds, upper_bounds)

# Convert bounds back from logarithmic scale to original scale for printing
lower_bounds_original = np.exp(lower_bounds)
upper_bounds_original = np.exp(upper_bounds)

# Print bounds
print("Lower bounds:", lower_bounds_original)
print("Upper bounds:", upper_bounds_original)

def fcost_log(params_log, sims, PD_data):
    params = np.exp(params_log.copy())
    return fcost(params, sims, PD_data)     

def callback_log(x, file_name='M1-temp'):
    callback(np.exp(x), file_name=file_name)

def callback_M1_evolution_log(x,convergence):
    callback_log(x, file_name='M1-temp-evolution')


acceptable_params_M1 = [] # Initiate an empty list to store all parameters that are below the chi2 limit


def fcost_uncertainty_M1(param_log, model, PD_data):
    global acceptable_params_M1

    params = np.exp(param_log) 
    cost = fcost(params, model, PD_data)

    dgf = sum(np.count_nonzero(np.isfinite(PD_data[exp]["SEM"])) for exp in PD_data)
    chi2_limit = chi2.ppf(0.95, dgf)

    if cost < chi2_limit:
        acceptable_params_M1.append(params)

    return cost

for i in range(0,5):
    niter = 0
    res = differential_evolution(func=fcost_uncertainty_M1, bounds=bounds_M1_log, args=args_M1, x0=params_M1_log, callback=callback_M1_evolution_log, disp = True) # This is the optimization
    params_M1_log = res['x'] # update starting parameters 


print(f"Number of parameter sets collected: {len(acceptable_params_M1)}") # Prints now many parameter sets that were accepted

# Save the accepted parameters to a csv file
with open('acceptable_params_M1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(acceptable_params_M1)


def plot_uncertainty(all_params, sims, PD_data, color='b', n_params_to_plot=500):
    import warnings
    random.shuffle(all_params)

    for experiment in PD_data:
        print(f"\nPlotting uncertainty for: {experiment}")
        plt.figure()  # Create a new figure
        timepoints = time_vectors[experiment]  # Time vector

        success_count = 0
        fail_count = 0

        for param in all_params:
            if success_count >= n_params_to_plot:
                break
            try:
                plot_sim(param, sims[experiment], timepoints, color)
                success_count += 1
            except RuntimeError as e:
                if "CVODE" in str(e):
                    fail_count += 1
                    continue  # Skip bad simulations
                else:
                    raise e  # Re-raise if it's not a CVODE issue

        # Plot the observed data on top
        plot_PD_dataset(PD_data[experiment])
        plt.title(experiment)
        print(f"  Successful simulations: {success_count}")
        print(f"  Failed (CVODE) simulations: {fail_count}")


p_opt_M1 = res['x']

# Plot the uncertainty
plot_uncertainty(acceptable_params_M1, first_model_sims, PD_data)

print(f"Chi2 limit: {chi2_limit}")

print(f"Params_m1_not_log: {np.exp(params_M1_log)}")

plt.show()