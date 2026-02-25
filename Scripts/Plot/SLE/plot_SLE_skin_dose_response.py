# Import necessary libraries
import sys
import os
import json
import numpy as np
import sund
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the best parameters
with open("../../../Results/Acceptable params/best_result_1_pdc_mm2.json", 'r') as f:
    params_1 = np.array(json.load(f)['best_param'])

with open("../../../Results/Acceptable params/best_result_10_pdc_mm2.json", 'r') as f:
    params_10 = np.array(json.load(f)['best_param'])

with open("../../../Results/Acceptable params/best_result_80_pdc_mm2.json", 'r') as f:
    params_80 = np.array(json.load(f)['best_param'])

with open("../../../Results/Acceptable params/best_result_400_pdc_mm2.json", 'r') as f:
    params_400 = np.array(json.load(f)['best_param'])

# Load acceptable parameters
acceptable_params_1 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()

acceptable_params_10 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()

acceptable_params_80 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()

acceptable_params_400 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL.csv", delimiter=",").tolist()

# Load dose response data
with open("../../../Data/SLE_skin_dose_response_1_pdc_mm2.json", "r") as f:
    dose_response_1 = json.load(f)

with open("../../../Data/SLE_skin_dose_response_10_pdc_mm2.json", "r") as f:
    dose_response_10 = json.load(f)

with open("../../../Data/SLE_skin_dose_response_80_pdc_mm2.json", "r") as f:
    dose_response_80 = json.load(f)

with open("../../../Data/SLE_skin_dose_response_400_pdc_mm2.json", "r") as f:
    dose_response_400 = json.load(f)

# Load SLE PK data from phase 2A
with open("../../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data = json.load(f)

# Load the mPBPK_SLE_models
with open("../../../Models/mPBPK_SLE_model_1_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

with open("../../../Models/mPBPK_SLE_model_10_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

with open("../../../Models/mPBPK_SLE_model_80_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

with open("../../../Models/mPBPK_SLE_model_400_pdc_mm2.txt", "r") as f:
    lines = f.readlines()

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model_1_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_10_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_80_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_400_pdc_mm2.txt')

model_1 = sund.load_model("mPBPK_SLE_model_1_pdc_mm2")
model_10 = sund.load_model("mPBPK_SLE_model_10_pdc_mm2")
model_80 = sund.load_model("mPBPK_SLE_model_80_pdc_mm2")
model_400 = sund.load_model("mPBPK_SLE_model_400_pdc_mm2")
# Assumed bodyweight for a fictional SLE patient 
bodyweight = 70

# Creating activity objects for each dose
IV_005_SLE = sund.Activity(time_unit='h')
IV_005_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 50]))

IV_03_SLE = sund.Activity(time_unit='h')
IV_03_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 300]))

IV_1_SLE = sund.Activity(time_unit='h')
IV_1_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 1000]))

IV_3_SLE = sund.Activity(time_unit='h')
IV_3_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 3000]))

IV_10_SLE = sund.Activity(time_unit='h')
IV_10_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 10000]))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 20000]))

IV_40_SLE = sund.Activity(time_unit='h')
IV_40_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 40000]))

IV_60_SLE = sund.Activity(time_unit='h')
IV_60_SLE.add_output("piecewise_constant", "IV_in",  t=[0],  f=bodyweight*np.array([0, 60000]))

SC_50_SLE = sund.Activity(time_unit='h')
SC_50_SLE.add_output("piecewise_constant", "SC_in",  t = PK_data['SCdose_50_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_50_SLE']['input']['SC_in']['f'])

SC_150_SLE = sund.Activity(time_unit='h')
SC_150_SLE.add_output("piecewise_constant", "SC_in",  t = PK_data['SCdose_150_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_150_SLE']['input']['SC_in']['f'])

SC_450_SLE = sund.Activity(time_unit='h')
SC_450_SLE.add_output("piecewise_constant", "SC_in",  t = PK_data['SCdose_450_SLE']['input']['SC_in']['t'],  f = PK_data['SCdose_450_SLE']['input']['SC_in']['f'])

# Define the time vectors for each dose
time_vector_low_IV_doses = np.arange(-10, 200, 1)
time_vector_medium_IV_doses = np.arange(-10, 2000, 1)
time_vector_high_IV_doses = np.arange(-10, 4600, 1)
time_vector_IV_20 = np.arange(-10, 4200, 1)
time_vector_SC_50 = np.arange(-10, 6000, 1)
time_vector_SC_150 = np.arange(-10, 7000, 1)
time_vector_SC_450 = np.arange(-10, 8000, 1)
time_vector_AUC = np.arange(0, 2688, 1)

# Conversion factor for plotting time in weeks (keep simulations in hours)
HOURS_PER_WEEK = 24 * 7

# Define a function to plot PK and PD simulations in skin together
# Change BDCA2 baseline concentration in mPBPK_SLE_model.txt to create plots for different pDC densities in skin lesions
def plot_skin_PK_PD_simulations_together(params, acceptable_params, models, time_vectors):

    # Loop through each dose
    for (patient_label, params), acceptable_params, model in zip(best_param_sets.items(), acceptable_param_sets.values(), models.values(),):
        
        save_dir=f"../../../Results/SLE/Skin/Dose_response/{patient_label}"
        os.makedirs(save_dir, exist_ok=True)

        doses = ['IV_005_SLE', 'IV_03_SLE', 'IV_1_SLE', 'IV_3_SLE', 'IV_10_SLE', 'IV_20_SLE', 'IV_40_SLE', 'IV_60_SLE']
        dose_labels = ['0.05 mg/kg IV dose', '0.3 mg/kg IV dose', '1 mg/kg IV dose', '3 mg/kg IV dose', '10 mg/kg IV dose', '20 mg/kg IV dose', '40 mg/kg IV dose', '60 mg/kg IV dose']

        SLE_best_params = np.delete(params.copy(), [10,15])

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

            # Create axes and labels for PK and PD (plot time in weeks)
            ax1.set_xlabel('Time [Weeks]')
            ax1.set_ylabel('Free Litifilimab Skin Concentration (µg/mL)', color='b')
            ax1.spines['top'].set_visible(False)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Free BDCA2 Expression on pDCs (% Change from Baseline)', color='r')
            ax2.spines['top'].set_visible(False)

            if dose in ('IV_005_SLE', 'IV_03_SLE'):
                time_vector = time_vectors['low_dose']
            elif dose in ('IV_1_SLE', 'IV_3_SLE'):
                time_vector = time_vectors['medium_dose']
            else:
                time_vector = time_vectors['high_dose']

            # Create a plotting time vector in weeks (simulations keep using hours)
            plot_time_vector = time_vector / HOURS_PER_WEEK

            # Plot PK_sim_skin (left y-axis)
            y_pk_min = np.full_like(time_vector, 10000)
            y_pk_max = np.full_like(time_vector, -10000)

            # Calculate PK uncertainty range
            for acceptable_param in acceptable_params:
                SLE_params = np.delete(acceptable_param.copy(), [10,15])
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
                    y_sim = sims[dose].feature_data[:, 2]
                    y_pk_min = np.minimum(y_pk_min, y_sim)
                    y_pk_max = np.maximum(y_pk_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            # Plot PK uncertainty range
            ax1.fill_between(plot_time_vector, y_pk_min, y_pk_max, color='b', alpha=0.3)

            sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
            y_pk = sims[dose].feature_data[:, 2]

            # Plot PD_sim_skin (right y-axis)
            y_pd_min = np.full_like(time_vector, 10000)
            y_pd_max = np.full_like(time_vector, -10000)

            # Calculate PD uncertainty range
            for acceptable_param in acceptable_params:
                SLE_params = np.delete(acceptable_param.copy(), [10,15])
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
                    y_sim = sims[dose].feature_data[:, 3]
                    y_pd_min = np.minimum(y_pd_min, y_sim)
                    y_pd_max = np.maximum(y_pd_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            # Plot PD uncertainty range
            ax2.fill_between(plot_time_vector, y_pd_min, y_pd_max, color='r', alpha=0.3)
            y_pd = sims[dose].feature_data[:, 3]

            ax1.plot(plot_time_vector, y_pk, 'b-', label='PK simulation')
            ax1.tick_params(axis='y', labelcolor='b')
            ax2.plot(plot_time_vector, y_pd, 'r-', label='PD simulation')
            ax2.tick_params(axis='y', labelcolor='r')

            # Legends and title
            ax1.legend(loc = 'lower right')
            ax2.legend(loc = 'upper right')
            plt.title(f"Simulation of a {dose_label} for a patient with {patient_label} pDCs/mm²", fontweight='bold')
            plt.tight_layout()

            # Save the figure
            save_path = os.path.join(save_dir, f"{dose}_PK_PD_sim_{patient_label}_pDC_mm2.png")
            plt.savefig(save_path, dpi=600)
            plt.close()


def plot_skin_PK_simulations(params, acceptable_params, models, time_vectors):

    doses = ['IV_20_SLE', 'SC_50_SLE', 'SC_150_SLE', 'SC_450_SLE']
    dose_labels = ['20 mg/kg IV dose', '50 mg SC dose', '150 mg SC dose', '450 mg SC dose']

    for dose, dose_label, time_vector in zip(doses, dose_labels, time_vectors.values()):
        save_dir = '../../../Results/SLE/Skin/PK/Predictions'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 8))

        linestyles = ['--', '-', ':', '-.']
        colors = plt.cm.Blues(np.linspace(0.6, 0.9, 4))

        for (patient_label, params), model, acceptable_params, linestyle, color in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), linestyles, colors):

            SLE_best_params = np.delete(params.copy(), [10,15])

            sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
                    'SC_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
                    'SC_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
                    'SC_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
                    } 

            y_min = np.full_like(time_vector, 10000)
            y_max = np.full_like(time_vector, -10000)

            # plotting time vector in weeks
            plot_time_vector = time_vector / HOURS_PER_WEEK

            for acceptable_param in acceptable_params:
                SLE_params = np.delete(acceptable_param.copy(), [10,15])
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
                    y_sim = sims[dose].feature_data[:, 2]
                    y_min = np.minimum(y_min, y_sim)
                    y_max = np.maximum(y_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            # Plot uncertainty range (x in weeks)
            plt.fill_between(plot_time_vector, y_min, y_max, color=color, alpha=0.5)

            sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
            y = sims[dose].feature_data[:, 2]
            plt.plot(plot_time_vector, y, color=color, label=f"{patient_label} pDCs/mm²", linestyle=linestyle, linewidth=2)

        plt.xlabel('Time [Weeks]', fontsize = 18)
        plt.ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize = 18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.yscale('log')
        plt.title('PK Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'upper right')

        if dose == 'IV_20_SLE':
            plt.ylim(0.001, 100)
        elif dose == 'SC_50_SLE':
            plt.ylim(0.00001, 1)
        elif dose == 'SC_150_SLE':
            plt.ylim(0.00001, 10)
        elif dose == 'SC_450_SLE':
            plt.ylim(0.00001, 100)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"PK_skin_sim_{dose}_test.png")
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_skin_PD_simulations(params, acceptable_params, models, time_vectors):

    doses = ['IV_20_SLE', 'SC_50_SLE', 'SC_150_SLE', 'SC_450_SLE']
    dose_labels = ['20 mg/kg IV dose', '50 mg SC dose', '150 mg SC dose', '450 mg SC dose']

    for dose, dose_label, time_vector in zip(doses, dose_labels, time_vectors.values()):
        save_dir = '../../../Results/SLE/Skin/PD/Predictions'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 8))

        linestyles = ['--', '-', ':', '-.']
        colors = plt.cm.Reds(np.linspace(0.6, 0.9, 4))

        for (patient_label, params), model, acceptable_params, linestyle, color in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), linestyles, colors):   
            
            SLE_best_params = np.delete(params.copy(), [10,15])
            
            sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
                    'SC_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
                    'SC_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
                    'SC_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
                    }

            y_min = np.full_like(time_vector, 10000)
            y_max = np.full_like(time_vector, -10000)

            # plotting time vector in weeks
            plot_time_vector = time_vector / HOURS_PER_WEEK

            for acceptable_param in acceptable_params:
                SLE_params = np.delete(acceptable_param.copy(), [10,15])
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
                    y_sim = sims[dose].feature_data[:, 3]
                    y_min = np.minimum(y_min, y_sim)
                    y_max = np.maximum(y_max, y_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e

            # Plot uncertainty range (x in weeks)
            plt.fill_between(plot_time_vector, y_min, y_max, color=color, alpha=0.5)

            sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
            y = sims[dose].feature_data[:, 3]
            plt.plot(plot_time_vector, y, color=color, label=f"{patient_label} pDCs/mm²", linestyle=linestyle, linewidth=2)

            plt.xlabel('Time [Weeks]', fontsize = 18)
            plt.ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize = 18)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.title('PD Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold')
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'lower right')
            plt.tight_layout()

        save_path = os.path.join(save_dir, f"PD_skin_sim_{dose}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()


def plot_dose_response_IV_doses_separate(dose_response_datasets):
    save_dir='../../../Results/SLE/Skin/Dose_response'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12,8))

    colors = plt.cm.Purples(np.linspace(0.6, 0.9, 4))
    linestyles = ['--', '-', ':', '-.']


    for (patient_label, dataset), color, linestyle in zip(dose_response_datasets.items(), colors, linestyles):

        plt.fill_between(dataset['IVdoses_SLE']['Dose'], dataset['IVdoses_SLE_lower']['Response'], dataset['IVdoses_SLE_higher']['Response'], color=color, alpha = 0.5) 
        plt.plot(dataset['IVdoses_SLE']['Dose'], dataset['IVdoses_SLE']['Response'],linestyle=linestyle, color = color, linewidth = 2, label = f"{patient_label} pDCs/mm²")

        plt.xlabel('IV Dose Size [mg/kg]', fontsize = 18)
        plt.ylabel('Response Time [weeks]', fontsize = 18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlim(0, 60)
        plt.ylim(0, 27)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title(f"Simulated Dose-Response in a 70 kg SLE Patient", fontsize = 22, fontweight='bold')
        plt.legend(loc = 'upper left', fontsize = 16, title = 'pDC Skin Density', title_fontsize = 18)
        plt.tight_layout()

    save_path = os.path.join(save_dir, f"Dose_response_pdc_density_separate.png")
    plt.savefig(save_path, dpi=600)
    plt.close()



def plot_skin_plasma_AUC_ratio(params, acceptable_params, models, time_vector):
    save_dir = '../../../Results/SLE/Skin/PK/Predictions'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12,8))

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

            # Sample up to 1000 acceptable parameter sets randomly (without replacement)
            try:
                acc_list = list(acceptable_params)
            except Exception:
                acc_list = []

            if len(acc_list) == 0:
                sampled_params = []
                print(f"No acceptable params for patient {patient_label}; skipping uncertainty for {dose_label}")
            else:
                rng = np.random.default_rng()
                n_draw = min(3000, len(acc_list))
                # choose indices without replacement
                indices = rng.choice(len(acc_list), size=n_draw, replace=False)
                sampled_params = [acc_list[i] for i in indices]

            for acceptable_param in sampled_params:
                SLE_params = np.delete(acceptable_param.copy(), [10,15])
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
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

            params = np.delete(params.copy(), [10,15])
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

    save_path = os.path.join(save_dir, "AUC_skin_plasma_ratios_test.txt")
    with open(save_path, "w") as f:   
    
        f.write(f"{'Patient':<8}{'Dose':<12}{'AUC_ratio':>12}{'AUC_min':>12}{'AUC_max':>12}\n")
        for row in results:  
            f.write(f"{row[0]:<8}{row[1]:<12}{row[2]:12.3f}{row[3]:12.3f}{row[4]:12.3f}\n")

    patient_labels = ['1', '10', '80', '400']
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

    plt.figure(figsize=(12, 8))
    bar_width = 0.2
    x = np.arange(len(doses))
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, 4))

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

    plt.xticks(x + 1.5*bar_width, dose_sizes, fontsize=16)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.ylabel('AUC Skin/Plasma Ratio [%]', fontsize=18)
    plt.xlabel('IV Dose Size [mg/kg]', fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yscale('log')
    plt.ylim(0.001, 100)
    plt.suptitle('Simulated AUC Skin/Plasma Ratio of Litifilimab Concentration', fontsize=22, fontweight='bold')
    plt.title('Measured over 16 Weeks for a 70 kg SLE Patient', fontsize=18)
    plt.legend(title='pDC Density', title_fontsize=18, fontsize=16)
    plt.tight_layout()

    bar_save_path = os.path.join(save_dir, "AUC_skin_plasma_ratios_barplot_test.png")
    plt.savefig(bar_save_path, dpi=600)
    plt.close()


models = {'1': model_1, '10': model_10, '80': model_80, '400': model_400}
best_param_sets = {'1': params_1, '10': params_10, '80': params_80, '400': params_400}
acceptable_param_sets = {'1': acceptable_params_1, '10': acceptable_params_10, '80': acceptable_params_80, '400': acceptable_params_400}
time_vectors_IV = {'low_dose': time_vector_low_IV_doses, 'medium_dose': time_vector_medium_IV_doses, 'high_dose': time_vector_high_IV_doses}


dose_response_datasets = {'1': dose_response_1, '10': dose_response_10, '80': dose_response_80, '400': dose_response_400}
time_vectors_IV_SC = {'IV_20': time_vector_IV_20, 'SC_50': time_vector_SC_50, 'SC_150': time_vector_SC_150, 'SC_450': time_vector_SC_450}

# plot_skin_PK_PD_simulations_together(best_param_sets, acceptable_param_sets, models, time_vectors_IV)

# plot_dose_response_IV_doses_separate(dose_response_datasets)

plot_skin_PK_simulations(best_param_sets, acceptable_param_sets, models, time_vectors_IV_SC)

# plot_skin_PD_simulations(best_param_sets, acceptable_param_sets, models, time_vectors_IV_SC)

# plot_skin_plasma_AUC_ratio(best_param_sets, acceptable_param_sets, models, time_vector_AUC)

