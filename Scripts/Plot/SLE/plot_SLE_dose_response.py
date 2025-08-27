# Import necessary libraries
import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt

# Load the best parameters for SLE
with open("../../../Results/Acceptable params/best_SLE_result.json", 'r') as f:
    params = np.array(json.load(f)['best_param'])

# Load acceptable parameters for SLE
with open("../../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params = json.load(f)

with open("../../../Data/SLE_skin_dose_response_32_pdc_mm2.json", "r") as f:
    dose_response_32 = json.load(f)

with open("../../../Data/SLE_skin_dose_response_179_pdc_mm2.json", "r") as f:
    dose_response_179 = json.load(f)

# Load the mPBPK_SLE_model
with open("../../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
model = sund.load_model("mPBPK_SLE_model")

# Assumed bodyweight for a fictional SLE patient 
bodyweight = 70

# Creating activity objects for each dose
# IV_005_SLE = sund.Activity(time_unit='h')
# IV_005_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 50]))

# IV_03_SLE = sund.Activity(time_unit='h')
# IV_03_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 300]))

# IV_1_SLE = sund.Activity(time_unit='h')
# IV_1_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 1000]))

# IV_3_SLE = sund.Activity(time_unit='h')
# IV_3_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 3000]))

# IV_10_SLE = sund.Activity(time_unit='h')
# IV_10_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 10000]))

# IV_20_SLE = sund.Activity(time_unit='h')
# IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 20000]))

# IV_40_SLE = sund.Activity(time_unit='h')
# IV_40_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 40000]))

# IV_60_SLE = sund.Activity(time_unit='h')
# IV_60_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 60000]))

SC_1400_SLE = sund.Activity(time_unit='h')
SC_1400_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = [0],  f = [0, 1400000])


model_sims = {
    # 'IV_005_SLE': sund.Simulation(models=model, activities=IV_005_SLE, time_unit='h'),
    # 'IV_03_SLE': sund.Simulation(models=model, activities=IV_03_SLE, time_unit='h'),
    # 'IV_1_SLE': sund.Simulation(models=model, activities=IV_1_SLE, time_unit='h'),
    # 'IV_3_SLE': sund.Simulation(models=model, activities=IV_3_SLE, time_unit='h'),
    # 'IV_10_SLE': sund.Simulation(models=model, activities=IV_10_SLE, time_unit='h'),
    # 'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
    # 'IV_40_SLE': sund.Simulation(models=model, activities=IV_40_SLE, time_unit='h'),
    # 'IV_60_SLE': sund.Simulation(models=model, activities=IV_60_SLE, time_unit='h'),
    'SC_1400_SLE': sund.Simulation(models=model, activities=SC_1400_SLE, time_unit='h')

}

# Define the time vectors for each dose
time_vector = np.arange(-10, 5000, 1)

# Define a function to plot PK and PD simulations in skin together
# Change BDCA2 baseline concentration in mPBPK_SLE_model.txt to create plots for different pDC densities in skin lesions
def plot_PK_PD_sim_with_data(params, acceptable_params, sims, time_vector, save_dir='../../../Results/SLE/Skin/Dose_response'):
    os.makedirs(save_dir, exist_ok=True)

    # labels = ['0.05 mg/kg IV dose', '0.3 mg/kg IV dose', '1 mg/kg IV dose', '3 mg/kg IV dose', '10 mg/kg IV dose', '20 mg/kg IV dose', '40 mg/kg IV dose', '60 mg/kg IV dose']
    # doses = ['IV_005_SLE', 'IV_03_SLE', 'IV_1_SLE', 'IV_3_SLE', 'IV_10_SLE', 'IV_20_SLE', 'IV_40_SLE', 'IV_60_SLE']
    labels = ['1400 mg SC dose']
    doses = ['SC_1400_SLE']

    # Loop through each dose
    for dose, label in zip (doses, labels):
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # Create axes and labels for PK and PD
        ax1.set_xlabel('Time [Hours]')
        ax1.set_ylabel('Free Litifilimab Skin Concentration (µg/mL)', color='b')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Free BDCA2 Expression on pDCs (% Change from Baseline)', color='r')

        # Plot PK_sim_skin (left y-axis)
        y_min = np.full_like(time_vector, 10000)
        y_max = np.full_like(time_vector, -10000)

        # Calculate uncertainty range
        for param in acceptable_params:
            try:
                sims[dose].simulate(time_vector=time_vector, parameter_values=param, reset=True)
                y_sim = sims[dose].feature_data[:, 2]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {label}")
                else:
                    raise e

        # Plot uncertainty range
        ax1.fill_between(time_vector, y_min, y_max, color='b', alpha=0.3)


        sims[dose].simulate(time_vector=time_vector, parameter_values=params, reset=True)
        y = sims[dose].feature_data[:, 2]
        ax1.plot(time_vector, y, 'b-', label='PK simulation')
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot PD_sim_skin (right y-axis) 
        y_min = np.full_like(time_vector, 10000)
        y_max = np.full_like(time_vector, -10000)

        # Calculate uncertainty range
        for param in acceptable_params:
            try:
                sims[dose].simulate(time_vector=time_vector, parameter_values=param, reset=True)
                y_sim = sims[dose].feature_data[:, 3]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    print(f"Skipping unstable parameter set for {label}")
                else:
                    raise e

        # Plot uncertainty range
        ax2.fill_between(time_vector, y_min, y_max, color='r', alpha=0.3)


        y = sims[dose].feature_data[:, 3]
        ax2.plot(time_vector, y, 'r-', label='PD simulation')
        ax2.tick_params(axis='y', labelcolor='r')

        # Legends and title
        ax1.legend(bbox_to_anchor=(0.95, 0.60))
        ax2.legend(bbox_to_anchor=(0.95, 0.50))
        plt.title(f"Simulation of a {label} for a patient with 32 pDCs/mm²")
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(save_dir, f"{dose}_PK_PD_sim_32_pDC_mm2.png")
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_dose_response_relationship(dose_response_datasets, save_dir='../../../Results/SLE/Skin/Dose_response'):
    os.makedirs(save_dir, exist_ok=True)

    labels = ['32_pDCs_mm2', '179_pDCs_mm2']

    for (patient, dataset), label in zip(dose_response_datasets.items(), labels):
        plt.figure(figsize=(12,8))

        plt.fill_between(dataset['IVdoses_SLE']['Dose'], dataset['IVdoses_SLE_lower']['Response'], dataset['IVdoses_SLE_higher']['Response'], color='b', alpha = 0.3) 
        plt.plot(dataset['IVdoses_SLE']['Dose'], dataset['IVdoses_SLE']['Response'], color = 'b', linewidth = 2)

        plt.xlabel('IV dose size (mg/kg)')
        plt.ylabel('Response time (weeks)')
        plt.xlim(0, 60)
        plt.ylim(0, 20)
        plt.title(f"Dose-response relationship for a 70 kg SLE patient with {label}")

        save_path = os.path.join(save_dir, f"Dose_response_{label}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()

dose_response_datasets = {'32 pdcs/mm2': dose_response_32, '179 pdcs/mm2': dose_response_179}
        

# Plot PK and PD simulations in skin together
plot_PK_PD_sim_with_data(params, acceptable_params, model_sims, time_vector)

# plot_dose_response_relationship(dose_response_datasets)
