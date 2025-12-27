# Importing the necessary libraries
import os
import re
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import sund
import json

from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Open the mPBPK_SLE_model.txt file and read its contents
with open("../../../Models/mPBPK_SLE_model_80_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

# Load the SLE PD data for plotting
with open("../../../Data/SLE_PD_data_plotting.json", "r") as f:
    SLE_PD_data = json.load(f)

# Load final parameters for SLE
with open("../../../Models/final_parameters_SLE.json", "r") as f:
    best_params = json.load(f)

# Load the acceptable parameters
acceptable_params = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()


# Define a function to plot all doses with uncertainty
def plot_all_doses_with_uncertainty(best_params, acceptable_params, sims, SLE_PD_data, time_vectors, save_dir='../../../Results/SLE/Plasma/PD'):
    os.makedirs(save_dir, exist_ok=True)

    # Colors and markers for different doses
    dose_colors = ['#1b7837', '#01947b', '#628759', '#70b5aa', '#35978f', '#76b56e', '#6d65bf']
    doses = ['0.05 mg/kg IV Dose', '0.3 mg/kg IV Dose', '1 mg/kg IV Dose', '3 mg/kg IV Dose', '10 mg/kg IV Dose', '20 mg IV Dose', '50 mg SC Dose']

    # Loop through each experiment in SLE_PD_data
    for i, experiment in enumerate(SLE_PD_data):
        plt.figure(figsize=(12, 8))
        timepoints = time_vectors[experiment]
        color = dose_colors[i % len(dose_colors)]
        dose = doses[i % len(doses)]
        
        y_min = np.full_like(timepoints, 10000)
        y_max = np.full_like(timepoints, -10000)

        # Calculate uncertainty range
        for params in acceptable_params:
            SLE_params = np.delete(params.copy(), [10,15])
            try:
                sims[experiment].simulate(time_vector=timepoints, parameter_values=SLE_params, reset=True)
                y = sims[experiment].feature_data[:, 1]
                y_min = np.minimum(y_min, y)
                y_max = np.maximum(y_max, y)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {experiment}")
                else:
                    raise e

        # Plot uncertainty range (convert to weeks)
        hours_per_week = 168.0
        timepoints_weeks = timepoints / hours_per_week
        plt.fill_between(timepoints_weeks, y_min, y_max, color=color, alpha=0.3, label='Uncertainty')

        # Plot optimal simulation
        sims[experiment].simulate(time_vector=timepoints, parameter_values=best_params, reset=True)
        y = sims[experiment].feature_data[:, 1]
        plt.plot(timepoints_weeks, y, color=color, linestyle='-', label='Simulation', linewidth=3)

        # Plot the only available SLE data set
        if experiment == 'IVdose_20_SLE':
            exp_times_weeks = np.array(SLE_PD_data[experiment]['time']) / hours_per_week
            plt.errorbar(exp_times_weeks,
                         SLE_PD_data[experiment]['BDCA2_median'],
                         yerr=SLE_PD_data[experiment]['SEM'],
                         fmt='P',
                         color=color,
                         linestyle='None',
                         markersize=8,
                         capsize=4,
                         elinewidth=2,
                         label='Data')

        # Set plot labels and legend
        plt.xlabel('Time [Weeks]', fontsize=18)
        plt.ylabel('Total BDCA2 Expression on pDCs [% Change]', fontsize=18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title('PD Simulations in Plasma of SLE Patients', fontsize=22, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(title=f'{dose}', title_fontsize=18, fontsize=16, loc='lower right')

        if experiment == 'IVdose_20_SLE':
            plt.xlim(-200.0 / hours_per_week, 5400.0 / hours_per_week)
        elif experiment == 'IVdose_10_HV':
            plt.xlim(-170.0 / hours_per_week, 4800.0 / hours_per_week)
        elif experiment == 'IVdose_3_HV':
            plt.xlim(-140.0 / hours_per_week, 4100.0 / hours_per_week)
        elif experiment == 'IVdose_1_HV':
            plt.xlim(-120.0 / hours_per_week, 3050.0 / hours_per_week)
        elif experiment == 'IVdose_03_HV':
            plt.xlim(-90.0 / hours_per_week, 2250.0 / hours_per_week)
        elif experiment == 'IVdose_005_HV':
            plt.xlim(-50.0 / hours_per_week, 1400.0 / hours_per_week)
        elif experiment == 'SCdose_50_HV':
            plt.xlim(-100.0 / hours_per_week, 2750.0 / hours_per_week)

        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        if experiment == 'IVdose_20_SLE':
            plt.ylim(-110, 5)
        else:
            plt.ylim(-90, 5)
            
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"{experiment}_SLE_PD_plot.png")
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=600)


# Install the models
sund.install_model('../../../Models/mPBPK_SLE_model_80_pdc_mm2.txt')
print(sund.installed_models())

# Load the model objects
model = sund.load_model("mPBPK_SLE_model_80_pdc_mm2")

# Average bodyweight for SLE patients (cohort 8 in the phase 1 trial)
# Since HV plots are only for comparison, not to fit PD data, the same bodyweight is used
bodyweight = 69

# Creating activity objects for each dose
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output('piecewise_constant', "IV_in",  t = SLE_PD_data['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output('piecewise_constant', "IV_in",  t = SLE_PD_data['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output('piecewise_constant', "IV_in",  t = SLE_PD_data['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output('piecewise_constant', "IV_in",  t = SLE_PD_data['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output('piecewise_constant', "IV_in",  t = SLE_PD_data['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output('piecewise_constant', "IV_in",  t = SLE_PD_data['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(SLE_PD_data['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output('piecewise_constant', "SC_in",  t = SLE_PD_data['SCdose_50_HV']['input']['SC_in']['t'],  f = SLE_PD_data['SCdose_50_HV']['input']['SC_in']['f'])


# Creating simulation objects for each dose in SLE
model_sims = {
    'IVdose_005_HV': sund.Simulation(models=model, activities=IV_005_HV, time_unit='h'),
    'IVdose_03_HV': sund.Simulation(models=model, activities=IV_03_HV, time_unit='h'),
    'IVdose_1_HV': sund.Simulation(models=model, activities=IV_1_HV, time_unit='h'),
    'IVdose_3_HV': sund.Simulation(models=model, activities=IV_3_HV, time_unit='h'),
    'IVdose_10_HV': sund.Simulation(models=model, activities=IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
    'SCdose_50_HV': sund.Simulation(models=model, activities=SC_50_HV, time_unit='h')
}

# Define the time vectors for each experiment
time_vectors = {exp: np.arange(-400, SLE_PD_data[exp]["time"][-1] + 5000, 1) for exp in SLE_PD_data}

# Call the function to plot all doses with uncertainty
plot_all_doses_with_uncertainty(
    best_params=best_params,
    acceptable_params=acceptable_params,
    sims=model_sims,
    SLE_PD_data=SLE_PD_data,
    time_vectors=time_vectors
)

