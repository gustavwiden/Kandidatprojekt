# Import necessary libraries
import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the best parameters for SLE
with open("../../../Results/Acceptable params/best_SLE_result_10_pdc_mm2.json", 'r') as f:
    params_10 = np.array(json.load(f)['best_param'])

with open("../../../Results/Acceptable params/best_SLE_result_32_pdc_mm2.json", 'r') as f:
    params_32 = np.array(json.load(f)['best_param'])

with open("../../../Results/Acceptable params/best_SLE_result_179_pdc_mm2.json", 'r') as f:
    params_179 = np.array(json.load(f)['best_param'])

with open("../../../Results/Acceptable params/best_SLE_result_500_pdc_mm2.json", 'r') as f:
    params_500 = np.array(json.load(f)['best_param'])

# Load acceptable parameters for SLE
with open("../../../Results/Acceptable params/acceptable_params_SLE_10_pdc_mm2.json", "r") as f:
    acceptable_params_10 = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE_32_pdc_mm2.json", "r") as f:
    acceptable_params_32 = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE_179_pdc_mm2.json", "r") as f:
    acceptable_params_179 = json.load(f)

with open("../../../Results/Acceptable params/acceptable_params_SLE_500_pdc_mm2.json", "r") as f:
    acceptable_params_500 = json.load(f)

# Load dose response data
with open("../../../Data/SLE_skin_dose_response_10_pdc_mm2.json", "r") as f:
    dose_response_10 = json.load(f)

with open("../../../Data/SLE_skin_dose_response_32_pdc_mm2.json", "r") as f:
    dose_response_32 = json.load(f)

with open("../../../Data/SLE_skin_dose_response_179_pdc_mm2.json", "r") as f:
    dose_response_179 = json.load(f)

with open("../../../Data/SLE_skin_dose_response_500_pdc_mm2.json", "r") as f:
    dose_response_500 = json.load(f)

# Load SLE PK data from phase 2A
with open("../../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load the mPBPK_SLE_model
with open("../../../Models/mPBPK_SLE_model_10_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

with open("../../../Models/mPBPK_SLE_model_32_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

with open("../../../Models/mPBPK_SLE_model_179_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

with open("../../../Models/mPBPK_SLE_model_500_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model_10_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_32_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_179_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_500_pdc_mm2.txt')

model_10 = sund.load_model("mPBPK_SLE_model_10_pdc_mm2")
model_32 = sund.load_model("mPBPK_SLE_model_32_pdc_mm2")
model_179 = sund.load_model("mPBPK_SLE_model_179_pdc_mm2")
model_500 = sund.load_model("mPBPK_SLE_model_500_pdc_mm2")

# Assumed bodyweight for a fictional SLE patient 
bodyweight = 70

# Creating activity objects for each dose
IV_005_SLE = sund.Activity(time_unit='h')
IV_005_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 50]))

IV_03_SLE = sund.Activity(time_unit='h')
IV_03_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 300]))

IV_1_SLE = sund.Activity(time_unit='h')
IV_1_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 1000]))

IV_3_SLE = sund.Activity(time_unit='h')
IV_3_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 3000]))

IV_10_SLE = sund.Activity(time_unit='h')
IV_10_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 10000]))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 20000]))

IV_40_SLE = sund.Activity(time_unit='h')
IV_40_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 40000]))

IV_60_SLE = sund.Activity(time_unit='h')
IV_60_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t=[0],  f=bodyweight*np.array([0, 60000]))

SC_50_SLE = sund.Activity(time_unit='h')
SC_50_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_50_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_50_SLE']['input']['SC_in']['f'])

SC_150_SLE = sund.Activity(time_unit='h')
SC_150_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_150_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_150_SLE']['input']['SC_in']['f'])

SC_450_SLE = sund.Activity(time_unit='h')
SC_450_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data['SCdose_450_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_450_SLE']['input']['SC_in']['f'])

# Define the time vectors for each dose
time_vector_low_IV_doses = np.arange(-10, 200, 1)
time_vector_medium_IV_doses = np.arange(-10, 2000, 1)
time_vector_high_IV_doses = np.arange(-10, 4500, 1)
time_vector_multiple_SC_doses = np.arange(-10, 7000, 1)
time_vector_AUC = np.arange(0, 2688, 1)

# Define a function to plot PK and PD simulations in skin together
# Change BDCA2 baseline concentration in mPBPK_SLE_model.txt to create plots for different pDC densities in skin lesions
def plot_skin_PK_PD_simulations_together(params, acceptable_params, models, time_vectors):

    # Loop through each dose
    for (patient_label, params), acceptable_params, model in zip(best_param_sets.items(), acceptable_param_sets.values(), models.values(),):

        save_dir=f"../../../Results/SLE/Skin/Dose_response/{patient_label}"
        os.makedirs(save_dir, exist_ok=True)

        doses = ['IV_005_SLE', 'IV_03_SLE', 'IV_1_SLE', 'IV_3_SLE', 'IV_10_SLE', 'IV_20_SLE', 'IV_40_SLE', 'IV_60_SLE']
        dose_labels = ['0.05 mg/kg IV dose', '0.3 mg/kg IV dose', '1 mg/kg IV dose', '3 mg/kg IV dose', '10 mg/kg IV dose', '20 mg/kg IV dose', '40 mg/kg IV dose', '60 mg/kg IV dose']

        for dose, dose_label in zip(doses, dose_labels):
            fig, ax1 = plt.subplots(figsize=(8, 5))

            sims = {
            'IV_005_SLE': sund.Simulation(models=model, activities=IV_005_SLE, time_unit='h'),
            'IV_03_SLE': sund.Simulation(models=model, activities=IV_03_SLE, time_unit='h'),
            'IV_1_SLE': sund.Simulation(models=model, activities=IV_1_SLE, time_unit='h'),
            'IV_3_SLE': sund.Simulation(models=model, activities=IV_3_SLE, time_unit='h'),
            'IV_10_SLE': sund.Simulation(models=model, activities=IV_10_SLE, time_unit='h'),
            'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
            'IV_40_SLE': sund.Simulation(models=model, activities=IV_40_SLE, time_unit='h'),
            'IV_60_SLE': sund.Simulation(models=model, activities=IV_60_SLE, time_unit='h')}

            # Create axes and labels for PK and PD
            ax1.set_xlabel('Time [Hours]')
            ax1.set_ylabel('Free Litifilimab Skin Concentration (µg/mL)', color='b')
            ax2 = ax1.twinx()
            ax2.set_ylabel('Free BDCA2 Expression on pDCs (% Change from Baseline)', color='r')

            if dose in ('IV_005_SLE', 'IV_03_SLE'):
                time_vector = time_vectors['low_dose']
            elif dose in ('IV_1_SLE', 'IV_3_SLE'):
                time_vector = time_vectors['medium_dose']
            else:
                time_vector = time_vectors['high_dose'] 

            # Plot PK_sim_skin (left y-axis)
            y_min = np.full_like(time_vector, 10000)
            y_max = np.full_like(time_vector, -10000)

            # Calculate uncertainty range
            for acceptable_param in acceptable_params:
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=acceptable_param, reset=True)
                    y_sim = sims[dose].feature_data[:, 2]
                    y_min = np.minimum(y_min, y_sim)
                    y_max = np.maximum(y_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
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
            for acceptable_param in acceptable_params:
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=acceptable_param, reset=True)
                    y_sim = sims[dose].feature_data[:, 3]
                    y_min = np.minimum(y_min, y_sim)
                    y_max = np.maximum(y_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            # Plot uncertainty range
            ax2.fill_between(time_vector, y_min, y_max, color='r', alpha=0.3)


            y = sims[dose].feature_data[:, 3]
            ax2.plot(time_vector, y, 'r-', label='PD simulation')
            ax2.tick_params(axis='y', labelcolor='r')

            # Legends and title
            ax1.legend(loc = 'lower right')
            ax2.legend(loc = 'upper right')
            plt.title(f"Simulation of a {dose_label} for a patient with {patient_label} pDCs/mm²")
            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(save_dir, f"{dose}_PK_PD_sim_{patient_label}_pDC_mm2.png")
            plt.savefig(save_path, dpi=600)
            plt.close()



def plot_skin_PK_simulations(params, acceptable_params, models, time_vectors):

    doses = ['IV_20_SLE', 'SC_50_SLE', 'SC_150_SLE', 'SC_450_SLE']
    dose_labels = ['20 mg/kg IV dose', '50 mg SC dose', '150 mg SC dose', '450 mg SC dose']

    for dose, dose_label in zip(doses, dose_labels):
        save_dir = '../../../Results/SLE/Skin/PK/Predictions'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))

        if dose == 'IV_20_SLE':
            time_vector = time_vectors['IV_dose']
        else:
            time_vector = time_vectors['SC_dose']

        linestyles = [':', '-', '--', '-.']

        for (patient_label, params), model, acceptable_params, linestyle in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), linestyles):

            sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
                    'SC_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
                    'SC_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
                    'SC_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
                    } 

            y_min = np.full_like(time_vector, 10000)
            y_max = np.full_like(time_vector, -10000)

            for acceptable_param in acceptable_params:
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=acceptable_param, reset=True)
                    y_sim = sims[dose].feature_data[:, 2]
                    y_min = np.minimum(y_min, y_sim)
                    y_max = np.maximum(y_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            # Plot uncertainty range
            plt.fill_between(time_vector, y_min, y_max, color='b', alpha=0.3)

            sims[dose].simulate(time_vector=time_vector, parameter_values=params, reset=True)
            y = sims[dose].feature_data[:, 2]
            plt.plot(time_vector, y, color='b', label=f"{patient_label} pDCs/mm²", linestyle=linestyle,)

            plt.xlabel('Time [Hours]')
            plt.ylabel('Free Litifilimab Plasma Concentration (µg/ml)')
            plt.title('PK Simulations in Skin of SLE Patients')
            plt.legend( title = 'pDC Density in Skin Lesions')

        save_path = os.path.join(save_dir, f"PK_skin_sim_{dose}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_skin_PD_simulations(params, acceptable_params, models, time_vectors):

    patient_labels = ['10 pDCs/mm²', '32 pDCs/mm²', '179 pDCs/mm²', '500 pDCs/mm²']
    doses = ['IV_20_SLE', 'SC_50_SLE', 'SC_150_SLE', 'SC_450_SLE']
    dose_labels = ['20 mg/kg IV dose', '50 mg SC dose', '150 mg SC dose', '450 mg SC dose']

    for dose, dose_label in zip(doses, dose_labels):
        save_dir = '../../../Results/SLE/Skin/PD/Predictions'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(8, 5))

        if dose == 'IV_20_SLE':
            time_vector = time_vectors['IV_dose']
        else:
            time_vector = time_vectors['SC_dose']

        linestyles = [':', '-', '--', '-.']

        for (patient_label, params), model, acceptable_params, linestyle in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), linestyles):   
            sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
                    'SC_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
                    'SC_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
                    'SC_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
                    }

            y_min = np.full_like(time_vector, 10000)
            y_max = np.full_like(time_vector, -10000)

            for acceptable_param in acceptable_params:
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=acceptable_param, reset=True)
                    y_sim = sims[dose].feature_data[:, 3]
                    y_min = np.minimum(y_min, y_sim)
                    y_max = np.maximum(y_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            # Plot uncertainty range
            plt.fill_between(time_vector, y_min, y_max, color='r', alpha=0.3)

            sims[dose].simulate(time_vector=time_vector, parameter_values=params, reset=True)
            y = sims[dose].feature_data[:, 3]
            plt.plot(time_vector, y, color='r', label=f"{patient_label} pDCs/mm²", linestyle=linestyle,)

            plt.xlabel('Time [Hours]')
            plt.ylabel('Free BDCA2 Expression on pDCs (% Change from Baseline)')
            plt.title('PD Simulations in Skin of SLE Patients')
            plt.legend(title = 'pDC Density in Skin Lesions')

        save_path = os.path.join(save_dir, f"PD_skin_sim_{dose}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_dose_response_IV_doses_separate(dose_response_datasets):
    save_dir='../../../Results/SLE/Skin/Dose_response'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12,8))

    colors = ['r', 'b', 'y', 'g']


    for (patient_label, dataset), color in zip(dose_response_datasets.items(), colors):

        plt.fill_between(dataset['IVdoses_SLE']['Dose'], dataset['IVdoses_SLE_lower']['Response'], dataset['IVdoses_SLE_higher']['Response'], color=color, alpha = 0.3) 
        plt.plot(dataset['IVdoses_SLE']['Dose'], dataset['IVdoses_SLE']['Response'], color = color, linewidth = 2, label = f"{patient_label} pDCs/mm²")

        plt.xlabel('IV dose size (mg/kg)', fontsize = 16)
        plt.ylabel('Response time (weeks)', fontsize = 16)
        plt.xlim(0, 60)
        plt.ylim(0, 27)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(f"Simulated Dose-Response in a 70 kg SLE Patient", fontsize = 20)
        plt.legend(loc = 'upper left', fontsize = 14, title = 'pDC Density in Skin Lesions', title_fontsize = 15)

    save_path = os.path.join(save_dir, f"Dose_response_pdc_density_separate.png")
    plt.savefig(save_path, dpi=600)
    plt.close()


def plot_dose_response_IV_doses_gradient(dose_response_datasets):
    save_dir='../../../Results/SLE/Skin/Dose_response'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12,8))

    linestyles = [':', '-', '--', '-.']

    highest_pdc_dataset = dose_response_datasets['10']
    lowest_pdc_dataset = dose_response_datasets['500']

    dose_size = np.asarray(highest_pdc_dataset['IVdoses_SLE']['Dose'], float)
    highest_response = np.asarray(highest_pdc_dataset['IVdoses_SLE_higher']['Response'], float)
    lowest_response = np.asarray(lowest_pdc_dataset['IVdoses_SLE_lower']['Response'], float)


    cm = LinearSegmentedColormap.from_list('Temperature Map', ['blue', 'red'])

    gradient = plt.imshow(np.linspace(0, 1, 256).reshape(-1, 1), extent=[dose_size.min(), dose_size.max(), lowest_response.min(), highest_response.max()],
                    origin='lower', aspect='auto', cmap=cm, alpha=0.8)

    poly = plt.fill_between(dose_size, lowest_response, highest_response, color='none')
    path = poly.get_paths()[0]
    gradient.set_clip_path(path, transform=plt.gca().transData)

    for (patient_label, dataset), linestyle in zip(dose_response_datasets.items(), linestyles):

        plt.plot(dataset['IVdoses_SLE']['Dose'], dataset['IVdoses_SLE']['Response'], color = 'k', linewidth = 3, linestyle = linestyle, label = f"{patient_label} pDCs/mm²")

        plt.xlabel('IV dose size (mg/kg)', fontsize = 16)
        plt.ylabel('Response time (weeks)', fontsize = 16)
        plt.xlim(0, 60)
        plt.ylim(0, 27)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(f"Simulated Dose-Response in a 70 kg SLE Patient", fontsize = 20)
        plt.legend(loc = 'upper left', fontsize = 14, title = 'pDC Density in Skin Lesions', title_fontsize = 15)

    save_path = os.path.join(save_dir, f"Dose_response_pdc_density_gradient.png")
    plt.savefig(save_path, dpi=600)
    plt.close()


def plot_skin_plasma_AUC_ratio(params, acceptable_params, models, time_vector):
    save_dir = '../../../Results/SLE/Skin/PK/Predictions'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12,6))

    doses = ['IV_005_SLE', 'IV_03_SLE', 'IV_1_SLE', 'IV_3_SLE', 'IV_10_SLE', 'IV_20_SLE', 'IV_40_SLE', 'IV_60_SLE']
    dose_labels = ['0.05 mg/kg IV dose', '0.3 mg/kg IV dose', '1 mg/kg IV dose', '3 mg/kg IV dose', '10 mg/kg IV dose', '20 mg/kg IV dose', '40 mg/kg IV dose', '60 mg/kg IV dose']

    results = []

    for dose, dose_label in zip(doses, dose_labels):
        for (patient_label, params), model, acceptable_params in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values()):
            sims = {
                'IV_005_SLE': sund.Simulation(models=model, activities=IV_005_SLE, time_unit='h'),
                'IV_03_SLE': sund.Simulation(models=model, activities=IV_03_SLE, time_unit='h'),
                'IV_1_SLE': sund.Simulation(models=model, activities=IV_1_SLE, time_unit='h'),
                'IV_3_SLE': sund.Simulation(models=model, activities=IV_3_SLE, time_unit='h'),
                'IV_10_SLE': sund.Simulation(models=model, activities=IV_10_SLE, time_unit='h'),
                'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
                'IV_40_SLE': sund.Simulation(models=model, activities=IV_40_SLE, time_unit='h'),
                'IV_60_SLE': sund.Simulation(models=model, activities=IV_60_SLE, time_unit='h')
            }

            plasma_min = np.full_like(time_vector, 10000)
            plasma_max = np.full_like(time_vector, -10000)
            skin_min = np.full_like(time_vector, 10000)
            skin_max = np.full_like(time_vector, -10000)

            for acceptable_param in acceptable_params:
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=acceptable_param, reset=True)
                    plasma_sim = sims[dose].feature_data[:, 0]
                    skin_sim = sims[dose].feature_data[:, 2]
                    plasma_min = np.minimum(plasma_min, plasma_sim)
                    plasma_max = np.maximum(plasma_max, plasma_sim)
                    skin_min = np.minimum(skin_min, skin_sim)
                    skin_max = np.maximum(skin_max, skin_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            sims[dose].simulate(time_vector=time_vector, parameter_values=params, reset=True)
            plasma_sim = sims[dose].feature_data[:, 0]
            skin_sim = sims[dose].feature_data[:, 2]

            AUC_plasma = np.trapezoid(plasma_sim, time_vector)
            AUC_skin = np.trapezoid(skin_sim, time_vector)
            AUC_plasma_min = np.trapezoid(plasma_min, time_vector)
            AUC_plasma_max = np.trapezoid(plasma_max, time_vector)
            AUC_skin_min = np.trapezoid(skin_min, time_vector)
            AUC_skin_max = np.trapezoid(skin_max, time_vector)

            AUC_ratio = 100 * AUC_skin / AUC_plasma if AUC_plasma != 0 else np.nan
            AUC_min_possible_ratio = 100 * AUC_skin_min / AUC_plasma_max if AUC_plasma_max != 0 else np.nan
            AUC_max_possible_ratio = 100 * AUC_skin_max / AUC_plasma_min if AUC_plasma_min != 0 else np.nan

            results.append([patient_label, dose, AUC_ratio, AUC_min_possible_ratio, AUC_max_possible_ratio])

    save_path = os.path.join(save_dir, "AUC_skin_plasma_ratios.txt")
    with open(save_path, "w") as f:   
    
        f.write(f"{'Patient':<8}{'Dose':<12}{'AUC_ratio':>12}{'AUC_min':>12}{'AUC_max':>12}\n")
        for row in results:  
            f.write(f"{row[0]:<8}{row[1]:<12}{row[2]:12.3f}{row[3]:12.3f}{row[4]:12.3f}\n")

    patient_labels = ['10', '32', '179', '500']
    dose_sizes = ['0.05 mg/kg', '0.3 mg/kg', '1 mg/kg', '3 mg/kg', '10 mg/kg', '20 mg/kg', '40 mg/kg', '60 mg/kg']
    
    auc_matrix = np.full((len(doses), len(patient_labels)), np.nan)
    auc_min_matrix = np.full((len(doses), len(patient_labels)), np.nan)
    auc_max_matrix = np.full((len(doses), len(patient_labels)), np.nan)
    for row in results:
        patient_idx = patient_labels.index(str(row[0]))
        dose_idx = doses.index(row[1])
        auc_matrix[dose_idx, patient_idx] = row[2]
        auc_min_matrix[dose_idx, patient_idx] = row[3]
        auc_max_matrix[dose_idx, patient_idx] = row[4]

    yerr = np.array([auc_matrix - auc_min_matrix, auc_max_matrix - auc_matrix])

    plt.figure(figsize=(14, 7))
    bar_width = 0.2
    x = np.arange(len(doses))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, patient_label in enumerate(patient_labels):
        plt.bar(
            x + i*bar_width,
            auc_matrix[:, i],
            width=bar_width,
            color=colors[i],
            label=f"{patient_label} pDCs/mm²",
            yerr=[auc_matrix[:, i] - auc_min_matrix[:, i], auc_max_matrix[:, i] - auc_matrix[:, i]],
            capsize=6
        )

    plt.xticks(x + 1.5*bar_width, dose_sizes, fontsize=14)
    plt.ylabel('AUC Ratio Skin/Plasma (%)', fontsize=16)
    plt.suptitle('Simulated AUC Ratio of Skin to Plasma Litifilimab Concentration over 16 Weeks', fontsize=22)
    plt.title('Following Single IV Doses in a 70 kg SLE Patient', fontsize=16)
    plt.legend(title='pDC Density', title_fontsize=16, fontsize=14)
    plt.tight_layout()

    bar_save_path = os.path.join(save_dir, "AUC_skin_plasma_ratios_barplot.png")
    plt.savefig(bar_save_path, dpi=300)
    plt.close()


models = {'10': model_10, '32': model_32, '179': model_179, '500': model_500}
best_param_sets = {'10': params_10, '32': params_32, '179': params_179, '500': params_500}
acceptable_param_sets = {'10': acceptable_params_10, '32': acceptable_params_32, '179': acceptable_params_179, '500': acceptable_params_500}
time_vectors_IV = {'low_dose': time_vector_low_IV_doses, 'medium_dose': time_vector_medium_IV_doses, 'high_dose': time_vector_high_IV_doses}


dose_response_datasets = {'10': dose_response_10, '32': dose_response_32, '179': dose_response_179, '500': dose_response_500}

time_vectors_IV_SC = {'IV_dose' : time_vector_high_IV_doses, 'SC_dose': time_vector_multiple_SC_doses}

# plot_skin_PK_PD_simulations_together(best_param_sets, acceptable_param_sets, models, time_vectors_IV)

# plot_dose_response_IV_doses_separate(dose_response_datasets)

# plot_dose_response_IV_doses_gradient(dose_response_datasets)

# plot_skin_PK_simulations(best_param_sets, acceptable_param_sets, models, time_vectors_IV_SC)

# plot_skin_PD_simulations(best_param_sets, acceptable_param_sets, models, time_vectors_IV_SC)

plot_skin_plasma_AUC_ratio(best_param_sets, acceptable_param_sets, models, time_vector_AUC)

