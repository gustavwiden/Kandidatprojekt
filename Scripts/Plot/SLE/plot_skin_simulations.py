# Import necessary libraries
import sys
import os
import json

from matplotlib import colors
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
acceptable_params_1 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL_1_pdc_mm2.csv", delimiter=",").tolist()

acceptable_params_10 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL_10_pdc_mm2.csv", delimiter=",").tolist()

acceptable_params_80 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL_80_pdc_mm2.csv", delimiter=",").tolist()

acceptable_params_400 = np.loadtxt("../../../Results/Acceptable params/acceptable_params_PL_400_pdc_mm2.csv", delimiter=",").tolist()


with open("../../../Data/SLE_skin_SC_dose_response_data.json", "r") as f:
    SC_dose_response_data = json.load(f)

# Load dose response data
with open("../../../Data/SLE_skin_IV_dose_response_data.json", "r") as f:
    IV_dose_response_data = json.load(f)

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

with open("../../../Models/mPBPK_model.txt", "r") as f:
    lines = f.readlines()

with open("../../../Models/mPBPK_model_high_plasma_pdc.txt", "r") as f:
    lines = f.readlines()

# Install the model
sund.install_model('../../../Models/mPBPK_SLE_model_1_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_10_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_80_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_SLE_model_400_pdc_mm2.txt')
sund.install_model('../../../Models/mPBPK_model.txt')
sund.install_model('../../../Models/mPBPK_model_high_plasma_pdc.txt')

model_1 = sund.load_model("mPBPK_SLE_model_1_pdc_mm2")
model_10 = sund.load_model("mPBPK_SLE_model_10_pdc_mm2")
model_80 = sund.load_model("mPBPK_SLE_model_80_pdc_mm2")
model_400 = sund.load_model("mPBPK_SLE_model_400_pdc_mm2")
model_HV = sund.load_model("mPBPK_model")
model_HV_high = sund.load_model("mPBPK_model_high_plasma_pdc")
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
time_vector_IV = np.arange(-10, 5100, 1)
time_vector_SC = np.arange(-10, 6800, 1)
time_vector_AUC = np.arange(0, 2688, 1)
time_vector_ratio = np.arange(0, 8000, 1)
time_vectors_IV_SC = {'IV': time_vector_IV, 'SC': time_vector_SC}

# Conversion factor for plotting time in weeks (keep simulations in hours)
hours_per_week = 24 * 7

# Define a function to plot PK and PD simulations in skin together
# Change BDCA2 baseline concentration in mPBPK_SLE_model.txt to create plots for different pDC densities in skin lesions
def plot_skin_PK_PD_simulations_together(best_param_sets, acceptable_param_sets, models, time_vector):

    response_results = {
        "1": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 40, 60], "Best": [], "Fast": [], "Slow": []},
        "10": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 40, 60], "Best": [], "Fast": [], "Slow": []},
        "80": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 40, 60], "Best": [], "Fast": [], "Slow": []},
        "400": {"Dose": [0.05, 0.3, 1, 3, 10, 20, 40, 60], "Best": [], "Fast": [], "Slow": []}
        }

    PK_colors = plt.cm.Blues(np.linspace(0.6, 0.9, 4))
    PD_colors = plt.cm.Reds(np.linspace(0.6, 0.9, 4))

    for (patient_label, best_params), acceptable_params, model, PK_color, PD_color in zip(best_param_sets.items(), acceptable_param_sets.values(), models.values(), PK_colors, PD_colors):
        save_dir=f"../../../Results/SLE/PKPD/{patient_label}"
        os.makedirs(save_dir, exist_ok=True)

        doses = ['IV_005_SLE', 'IV_03_SLE', 'IV_1_SLE', 'IV_3_SLE', 'IV_10_SLE', 'IV_20_SLE', 'IV_40_SLE', 'IV_60_SLE']
        dose_labels = ['0.05 mg/kg IV dose', '0.3 mg/kg IV dose', '1 mg/kg IV dose', '3 mg/kg IV dose', '10 mg/kg IV dose', '20 mg/kg IV dose', '40 mg/kg IV dose', '60 mg/kg IV dose']

        SLE_best_params = np.delete(best_params.copy(), [10,15])

        for dose, dose_label in zip(doses, dose_labels):
            fig, ax1 = plt.subplots(figsize=(10, 8))

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
            ax1.set_xlabel('Time [Weeks]', fontsize=18)
            ax1.set_ylabel('Free Litifilimab Skin Concentration (µg/mL)', color=PK_color, fontsize=18)
            ax1.spines['top'].set_visible(False)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Free BDCA2 Expression on pDCs (% Change from Baseline)', color=PD_color, fontsize=18)
            ax2.spines['top'].set_visible(False)

            # Create a plotting time vector in weeks (simulations keep using hours)
            time_weeks = time_vector / hours_per_week

            # Plot PK_sim_skin (left y-axis)
            y_pk_min = np.full_like(time_vector, 10000)
            y_pk_max = np.full_like(time_vector, -10000)
            y_pd_min = np.full_like(time_vector, 10000)
            y_pd_max = np.full_like(time_vector, -10000)

            # Calculate PK uncertainty range
            for acceptable_param in acceptable_params:
                SLE_params = np.delete(acceptable_param.copy(), [10,15])
                try:
                    sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
                    y_pk_sim = sims[dose].feature_data[:, 2]
                    y_pd_sim = sims[dose].feature_data[:, 3]
                    y_pk_min = np.minimum(y_pk_min, y_pk_sim)
                    y_pk_max = np.maximum(y_pk_max, y_pk_sim)
                    y_pd_min = np.minimum(y_pd_min, y_pd_sim)
                    y_pd_max = np.maximum(y_pd_max, y_pd_sim)
                except RuntimeError as e:
                    if "CV_ERR_FAILURE" in str(e):
                        print(f"Skipping unstable parameter set for {dose_label}")
                    else:
                        raise e
                    
            sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
            y_pk_best = sims[dose].feature_data[:, 2]
            y_pd_best = sims[dose].feature_data[:, 3]

            # Threshold is set to -95 % to show for how long the supression of BDCA2 on pDCs in skin is maintained.
            # For doses which never reach this threshold, the response time is set to None.
            response_threshold = -90

            def get_recovery_time(y_data):
                suppression_start_idx = np.where(y_data < response_threshold)[0] 
                if len(suppression_start_idx) == 0:
                    return 0 
                suppression_end_idx = np.where((y_data > response_threshold) & (time_weeks >= time_weeks[suppression_start_idx[0]]))[0]
                return round(float(time_weeks[suppression_end_idx[0]]), 2) if len(suppression_end_idx) > 0 else None

            response_results[patient_label]["Best"].append(get_recovery_time(y_pd_best))
            response_results[patient_label]["Fast"].append(get_recovery_time(y_pd_max))
            response_results[patient_label]["Slow"].append(get_recovery_time(y_pd_min))

            # Plot PK and PD uncertainty range
            ax1.fill_between(time_weeks, y_pk_min, y_pk_max, color=PK_color, alpha=0.5, label='PK Uncertainty')
            ax2.fill_between(time_weeks, y_pd_min, y_pd_max, color=PD_color, alpha=0.5, label='PD Uncertainty')
            
            ax1.plot(time_weeks, y_pk_best, color=PK_color, label='PK simulation')
            ax1.tick_params(axis='y', labelcolor=PK_color, which='major', labelsize=16)
            ax2.plot(time_weeks, y_pd_best, color=PD_color, label='PD simulation')
            ax2.tick_params(axis='y', labelcolor=PD_color, which='major', labelsize=16)

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()

            ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=16)

            plt.title(f"{dose_label} and pDC Density of {patient_label} pDCs/mm²", fontsize=18)
            plt.suptitle('PK and PD Simulations in Skin of a SLE Patient', fontsize=22, fontweight='bold')
            plt.tight_layout()

            # Save the figure
            save_path_png = os.path.join(save_dir, f"{dose}_PK_PD_sim_{patient_label}_pDC_mm2.png")
            plt.savefig(save_path_png, dpi=600)
            save_path_svg = os.path.join(save_dir, f"{dose}_PK_PD_sim_{patient_label}_pDC_mm2.svg")
            plt.savefig(save_path_svg)
            plt.close()

    # with open("../../../Data/SLE_skin_IV_dose_response_data.json", "w") as f:
    #     json.dump(response_results, f, indent=4)


def plot_skin_PK_simulations(best_param_sets, acceptable_param_sets, models, time_vectors):

    doses = ['IV_20_SLE', 'SC_50_SLE', 'SC_150_SLE', 'SC_450_SLE']
    dose_labels = ['For a Single 20 mg/kg IV dose', 'For Repeated 50 mg SC doses', 'For Repeated 150 mg SC doses', 'For Repeated 450 mg SC doses']

    for dose, dose_label in zip(doses, dose_labels):
        save_dir = '../../../Results/SLE/PK'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        if dose == 'IV_20_SLE':
            time_vector = time_vectors['IV']
        else:
            time_vector = time_vectors['SC']

        linestyles = ['--', '-', ':', '-.']
        colors = plt.cm.Blues(np.linspace(0.6, 0.9, 4))

        for (patient_label, best_params), model, acceptable_params, linestyle, color in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), linestyles, colors):
            SLE_best_params = np.delete(best_params.copy(), [10,15])

            sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
                    'SC_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
                    'SC_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
                    'SC_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
                    } 

            y_min = np.full_like(time_vector, 10000)
            y_max = np.full_like(time_vector, -10000)

            # plotting time vector in weeks
            time_weeks = time_vector / hours_per_week  

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
            plt.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.5)

            sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
            y = sims[dose].feature_data[:, 2]
            plt.plot(time_weeks, y, color=color, label=f"{patient_label} pDCs/mm²", linestyle=linestyle, linewidth=2)

        plt.xlabel('Time [Weeks]', fontsize = 18)
        plt.ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize = 18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.yscale('log')
        plt.title(f"{dose_label}", fontsize = 18)
        plt.suptitle('PK Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold', x=0.54)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'upper right')

        if dose == 'IV_20_SLE':
            plt.ylim(0.001, 100)
        else:
            plt.ylim(0.0001, 100)

        plt.tight_layout()

        save_path_png = os.path.join(save_dir, f"PK_skin_sim_{dose}.png")
        plt.savefig(save_path_png, dpi=600)
        save_path_svg = os.path.join(save_dir, f"PK_skin_sim_{dose}.svg")
        plt.savefig(save_path_svg)
        plt.close()


def plot_skin_PD_simulations(best_param_sets, acceptable_param_sets, models, time_vectors):

    doses = ['IV_20_SLE', 'SC_50_SLE', 'SC_150_SLE', 'SC_450_SLE']
    dose_labels = ['For a Single 20 mg/kg IV dose', 'For Repeated 50 mg SC doses', 'For Repeated 150 mg SC doses', 'For Repeated 450 mg SC doses']

    for dose, dose_label in zip(doses, dose_labels):
        save_dir = '../../../Results/SLE/PD'
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))

        if dose == 'IV_20_SLE':
            time_vector = time_vectors['IV']
        else:
            time_vector = time_vectors['SC']

        linestyles = ['--', '-', ':', '-.']
        colors = plt.cm.Reds(np.linspace(0.6, 0.9, 4))

        for (patient_label, best_params), model, acceptable_params, linestyle, color in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), linestyles, colors):   
            SLE_best_params = np.delete(best_params.copy(), [10,15])
            
            sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h'),
                    'SC_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
                    'SC_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
                    'SC_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
                    }

            y_min = np.full_like(time_vector, 10000)
            y_max = np.full_like(time_vector, -10000)

            # plotting time vector in weeks
            time_weeks = time_vector / hours_per_week

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
            plt.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.5)

            sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
            y = sims[dose].feature_data[:, 3]
            plt.plot(time_weeks, y, color=color, label=f"{patient_label} pDCs/mm²", linestyle=linestyle, linewidth=2)

            plt.xlabel('Time [Weeks]', fontsize = 18)
            plt.ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize = 18)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.title(f"{dose_label}", fontsize = 18)
            plt.suptitle('PD Simulations in Skin of SLE Patients', fontsize = 22, fontweight='bold', x=0.54)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(title = 'pDC Skin Density', title_fontsize = 18, fontsize = 16, loc = 'lower right')
            plt.tight_layout()

        save_path_png = os.path.join(save_dir, f"PD_skin_sim_{dose}.png")
        save_path_svg = os.path.join(save_dir, f"PD_skin_sim_{dose}.svg")
        plt.savefig(save_path_png, dpi=600)
        plt.savefig(save_path_svg)
        plt.close()


def plot_IV_dose_response(IV_dose_response_data):
    save_dir='../../../Results/SLE/Dose_response'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10,8))

    colors = plt.cm.Greens(np.linspace(0.6, 0.9, 4))
    linestyles = ['--', '-', ':', '-.']

    for (patient_label, dataset), color in zip(IV_dose_response_data.items(), colors):
        plt.fill_between(dataset['Dose'], dataset['Fast'], dataset['Slow'], color=color, alpha=0.4)

    for (patient_label, dataset), color, linestyle in zip(IV_dose_response_data.items(), colors, linestyles):
        plt.plot(dataset['Dose'], dataset['Best'], linestyle=linestyle, color=color, marker='o', linewidth=2, label=f"{patient_label} pDCs/mm²")

    plt.xlabel('IV Dose Size [mg/kg]', fontsize=18)
    plt.ylabel('Response Duration [Weeks]', fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlim(0, 60)
    plt.ylim(0, 27)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title("Continuous Suppression of >90% BDCA2 on pDCs in Skin", fontsize=18)
    plt.suptitle("Response Duration for a Single IV Dose", fontsize=22, fontweight='bold', x=0.52)
    plt.legend(loc='upper left', fontsize=16, title='pDC Skin Density', title_fontsize=18)
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(save_dir, "IV_dose_response.png"), dpi=600)
    plt.savefig(os.path.join(save_dir, "IV_dose_response.svg"))
    plt.close()


def plot_skin_plasma_AUC_ratio(best_param_sets, acceptable_param_sets, models, time_vector):
    save_dir = '../../../Results/SLE/PK'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10,8))

    doses = ['IV_005_SLE', 'IV_03_SLE', 'IV_1_SLE', 'IV_3_SLE', 'IV_10_SLE', 'IV_20_SLE', 'IV_40_SLE', 'IV_60_SLE']
    dose_labels = ['0.05 mg/kg IV dose', '0.3 mg/kg IV dose', '1 mg/kg IV dose', '3 mg/kg IV dose', '10 mg/kg IV dose', '20 mg/kg IV dose', '40 mg/kg IV dose', '60 mg/kg IV dose']

    results = []

    for dose, dose_label in zip(doses, dose_labels):
        for (patient_label, best_params), model, acceptable_params in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values()):
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

            SLE_best_params = np.delete(best_params.copy(), [10,15])
            sims[dose].simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
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

    patient_labels = ['1', '10', '80', '400']
    dose_sizes = ['0.05', '0.3', '1', '3', '10', '20', '40', '60']
    
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
    plt.ylabel('AUC Ratio Skin vs Plasma [%]', fontsize=18)
    plt.xlabel('IV Dose Size [mg/kg]', fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.yscale('log')
    plt.ylim(0.001, 100)
    plt.suptitle('AUC Ratio of Litifilimab in Skin vs Plasma', fontsize=22, fontweight='bold', x=0.52)
    plt.title('Simulated over 24 Weeks in SLE Patients', fontsize=18)
    plt.legend(title='pDC Skin Density', title_fontsize=18, fontsize=16)
    plt.tight_layout()

    bar_save_path_png = os.path.join(save_dir, "AUC_skin_plasma_ratios.png")
    plt.savefig(bar_save_path_png, dpi=600)
    bar_save_path_svg = os.path.join(save_dir, "AUC_skin_plasma_ratios.svg")
    plt.savefig(bar_save_path_svg)
    plt.close()


def plot_skin_plasma_concentration_ratio(best_param_sets, acceptable_param_sets, models, time_vector, save_dir='../../../Results/SLE/PK'):
    os.makedirs(save_dir, exist_ok=True)

    doses = ['IV_20_SLE', 'IV_20_SLE', 'IV_20_SLE', 'IV_20_SLE']
    labels = ["SLE Patient (10 pDCs/mm² in Skin)", "SLE Patient (80 pDCs/mm² in Skin)", "HV (12000 pDCs/mL in Blood)", "HV (5100 pDCs/mL in Blood)"]
    blue_shades= plt.cm.Blues(np.linspace(0.7, 0.9, 2))
    HV_color = blue_shades[0]
    SLE_color = blue_shades[1]
    colors = [SLE_color, SLE_color, HV_color, HV_color]
    linestyles = ['-', '--', '-', '--']
    

    fig, ax = plt.subplots(figsize=(10, 8))

    common_plasma_range = np.logspace(-3, 3, 500)

    timepoints = time_vector
    start_index = np.searchsorted(timepoints, 12)

    for (patient_label, best_params), model, acceptable_params, dose, color, label, linestyle in zip(best_param_sets.items(), models.values(), acceptable_param_sets.values(), doses, colors, labels, linestyles):
        sims = {'IV_20_SLE': sund.Simulation(models=model, activities=IV_20_SLE, time_unit='h')}

        all_interp_skin = []

        for acceptable_param in acceptable_params:
            if patient_label in ['HV', 'HV_high']:
                params = np.delete(acceptable_param.copy(), [11,16])
            else:
                params = np.delete(acceptable_param.copy(), [10,15])

            try:
                sims[dose].simulate(time_vector=timepoints, parameter_values=params, reset=True)
                plasma = sims[dose].feature_data[start_index:, 0]
                skin = sims[dose].feature_data[start_index:, 2]

                sort_idx = np.argsort(plasma)
                interp_skin = np.interp(common_plasma_range, plasma[sort_idx], skin[sort_idx], left=np.nan, right=np.nan)
                all_interp_skin.append(interp_skin)
            except RuntimeError:
                continue

        if all_interp_skin:
            y_min = np.nanmin(all_interp_skin, axis=0)
            y_max = np.nanmax(all_interp_skin, axis=0)
            plt.fill_between(common_plasma_range, y_min, y_max, color=color, alpha=0.3)

        if patient_label in ['HV', 'HV_high']:
            params_best = np.delete(best_params.copy(), [11,16])
        else:   
            params_best = np.delete(best_params.copy(), [10,15])
            
        sims[dose].simulate(time_vector=timepoints, parameter_values=params_best, reset=True)
        x_best_plasma = sims[dose].feature_data[start_index:, 0]
        y_best_skin = sims[dose].feature_data[start_index:, 2]

        if patient_label in ['HV', 'HV_high']:
            plt.plot(x_best_plasma, y_best_skin, color=color, label=f"{label}", linestyle=linestyle, linewidth=3)
        else:
            plt.plot(x_best_plasma, y_best_skin, color=color, label=f"{label}", linestyle=linestyle, linewidth=3)

    ref_x = np.logspace(-3, 3, 100) 
    plt.plot(ref_x, 0.157 * ref_x, 'k-', linewidth=3, label='Skin Distribution in Literature (15.7 %)')
    plt.plot(ref_x, 0.0785 * ref_x, 'k--', linewidth=2, label='2-fold Error')
    plt.plot(ref_x, 0.314 * ref_x, 'k--', linewidth=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-3, 4e2) 
    plt.ylim(1e-4, 1e2)
    plt.xlabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
    plt.ylabel('Free Litifilimab Skin Concentration [µg/ml]', fontsize=18)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(f'For a 20 mg/kg IV Dose in HV and SLE Patients', fontsize=18)
    plt.suptitle('Distribution of Litifilimab in Skin vs Plasma', fontsize=22, fontweight='bold', x=0.54)
    plt.legend(fontsize=16, loc='upper left')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    # Save the plot
    save_path_png = os.path.join(save_dir, f"Skin_vs_plasma_distribution.png")
    plt.savefig(save_path_png, format='png', dpi=600)
    save_path_svg = os.path.join(save_dir, "Skin_vs_plasma_distribution.svg")
    plt.savefig(save_path_svg, format='svg')
    plt.close()


def create_SC_activity(dose_mg, frequency_weeks, total_weeks, infusion_duration=5):
    """
    Generates a sund.Activity for repeated SC dosing.
    sund piecewise_constant requires len(f) = len(t) + 1.
    """
    dose_ug = dose_mg * 1000 
    hours_per_week = 168.0
    total_hours = total_weeks * hours_per_week
    interval_hours = frequency_weeks * hours_per_week
    
    dose_start_times = np.arange(0, total_hours, interval_hours)
    
    t_list = []
    f_list = []
    
    # First value in f is for the interval before t[0]
    f_list.append(0) 

    for start in dose_start_times:
        # At 'start', the infusion begins
        t_list.append(start)
        f_list.append(dose_ug)
        
        # At 'start + infusion_duration', the infusion ends
        t_list.append(start + infusion_duration)
        f_list.append(0) 
        
    activity = sund.Activity(time_unit='h')
    activity.add_output("piecewise_constant", "SC_in", t=t_list, f=f_list)
    
    # Maintenance ends one interval after the final injection
    maintenance_end_time = dose_start_times[-1] + interval_hours
    return activity, maintenance_end_time

def plot_PD_SC_frequency(best_param_sets, acceptable_param_sets, models, SC_dose_response_data):
    save_dir = '../../../Results/SLE/PD/SC_Frequency'
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration
    doses_mg = [50, 150, 300, 450, 600]
    treatment_duration_weeks = 25 
    threshold = -90
    hours_per_week = 168.0
    
    # Distinct colors and labels for our three critical scenarios
    scenarios = ['Fast', 'Best', 'Slow']
    colors = plt.cm.Reds(np.linspace(0.7, 0.9, 3)) 
    linestyles = [':', '-', '--']
    
    for density_label, best_params in best_param_sets.items():
        
        model = models[density_label]
        acceptable_params = acceptable_param_sets[density_label]
        SLE_best_params = np.delete(best_params.copy(), [10, 15])
        
        # Get the results found during the binary search for this density
        found_intervals = SC_dose_response_data[density_label]
        
        for dose_idx, dose_mg in enumerate(doses_mg):
            plt.figure(figsize=(10, 8))
            
            for s_idx, scenario in enumerate(scenarios):
                interval = found_intervals[scenario][dose_idx]
                
                if interval <= 0:
                    continue
                
                # INVERSION: Calculate doses per week
                frequency = round(1.0 / interval, 2)
                
                color = colors[s_idx]
                linestyle = linestyles[s_idx]
                
                # Create activity for this specific scenario interval
                activity, maintenance_end = create_SC_activity(dose_mg, interval, treatment_duration_weeks)
                
                time_vector = np.arange(0, maintenance_end + 2200, 1)
                time_weeks = time_vector / hours_per_week
                
                sim = sund.Simulation(models=model, activities=activity, time_unit='h')
                
                y_min = np.full_like(time_vector, 10000)
                y_max = np.full_like(time_vector, -10000)

                # Simulate Uncertainty
                for acceptable_param in acceptable_params:
                    SLE_params = np.delete(acceptable_param.copy(), [10, 15])
                    try:
                        sim.simulate(time_vector=time_vector, parameter_values=SLE_params, reset=True)
                        y_sim = sim.feature_data[:, 3]
                        y_min = np.minimum(y_min, y_sim)
                        y_max = np.maximum(y_max, y_sim)
                    except RuntimeError:
                        continue

                # Simulate and Plot the Best fit
                sim.simulate(time_vector=time_vector, parameter_values=SLE_best_params, reset=True)
                y_best = sim.feature_data[:, 3]
                
                # Updated Label to "x doses per week"
                label_text = f"{frequency} Doses/Week"
                
                plt.fill_between(time_weeks, y_min, y_max, color=color, alpha=0.2)
                plt.plot(time_weeks, y_best, color=color, linestyle=linestyle, 
                         linewidth=2, label=label_text)

            # Formatting
            plt.axhline(y=threshold, color='k', linestyle='--',alpha = 0.8, label='90% Threshold')
            plt.axvline(x=10, color='k', linestyle=':', alpha = 0.8, label='10-Week Threshold') 
            
            plt.xlabel('Time [Weeks]', fontsize=18)
            plt.ylabel('Free BDCA2 Expression on pDCs [% Change]', fontsize=18)
            plt.title(f"For Repeated {dose_mg} mg SC Doses and {density_label} pDCs/mm² in Skin", fontsize=18)
            plt.suptitle(f'PD Simulations in Skin of SLE Patient', fontsize=22, fontweight='bold', x=0.54)
            
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.legend(title='Frequency of SC Doses', title_fontsize=18, fontsize=16, loc='upper left')
            plt.tight_layout()

            plt.savefig(os.path.join(save_dir, f"PD_skin_sim_SC_frequency_{dose_mg}mg_{density_label}.png"), dpi=600)
            plt.savefig(os.path.join(save_dir, f"PD_skin_sim_SC_frequency_{dose_mg}mg_{density_label}.svg"))
            plt.close()

def check_suppression_maintained(y_data, time_vector, threshold, maintenance_end, start_week=10):
    """
    Checks if suppression is reached and maintained specifically between 
    week 10 and the end of the treatment period.
    """
    # Create mask for the maintenance window (Week 10 to End of Dosing)
    mask = (time_vector >= start_week * 168.0) & (time_vector <= maintenance_end)
    y_check = y_data[mask]
    
    if len(y_check) == 0:
        return False
    
    # 1. Must be suppressed at the end of the window
    is_suppressed_end = y_check[-1] <= threshold
    
    # 2. Crossing Criteria: Once in the maintenance window, it should not cross
    # back above the threshold (0 crossings if already suppressed, 1 if it dips in).
    crossings = np.sum(np.diff(np.sign(y_check - threshold)) != 0)
    
    return is_suppressed_end and crossings <= 1 

def simulate_SC_dose_response_frequency(density_label, best_param_sets, acceptable_param_sets, models):
    save_dir = '../../../Results/SLE/SC_Frequency'
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration
    doses_mg = [50, 150, 300, 450, 600]
    treatment_duration_weeks = 25 
    threshold = -90 
    hours_per_week = 168.0
    
    # Define the search bounds in hours
    # From 1 hour up to the full treatment duration (e.g., 60 weeks)
    MAX_HOURS = int(treatment_duration_weeks * hours_per_week)

    model = models[density_label]
    acceptable_params = acceptable_param_sets[density_label]
    SLE_best_params = np.delete(best_param_sets[density_label].copy(), [10, 15])
    
    results = {"Dose": doses_mg, "Best": [], "Fast": [], "Slow": []}
    
    for dose_mg in doses_mg:
        print(f"--- Searching Optimal Frequency for {dose_mg} mg (Density: {density_label}) ---")

        def evaluate_h_interval(h):
            """Returns (best_pass, fast_pass, slow_pass) for a given hour interval."""
            interval_w = h / hours_per_week
            
            # SAFETY: If the interval is shorter than the infusion duration, it's invalid
            if h <= 5: 
                return False, False, False
                
            activity, m_end = create_SC_activity(dose_mg, interval_w, treatment_duration_weeks)
            t_vec = np.arange(0, m_end + 1, 1)
            sim = sund.Simulation(models=model, activities=activity, time_unit='h')
            
            # Initialize pass flags
            pass_best, pass_fast, pass_slow = False, False, False
            
            # 1. Simulate Best with Error Handling
            try:
                sim.simulate(time_vector=t_vec, parameter_values=SLE_best_params, reset=True)
                pass_best = check_suppression_maintained(sim.feature_data[:, 3], t_vec, threshold, m_end)
            except RuntimeError:
                pass_best = False # Treat solver failure as a failure to maintain suppression

            # 2. Simulate Uncertainty with Error Handling
            y_min = np.full_like(t_vec, 10000)
            y_max = np.full_like(t_vec, -10000)
            sim_success_count = 0

            for acc_p in acceptable_params:
                p = np.delete(acc_p.copy(), [10, 15])
                try:
                    sim.simulate(time_vector=t_vec, parameter_values=p, reset=True)
                    y_sim = sim.feature_data[:, 3]
                    y_min = np.minimum(y_min, y_sim)
                    y_max = np.maximum(y_max, y_sim)
                    sim_success_count += 1
                except RuntimeError:
                    continue
            
            # Only evaluate uncertainty if at least some simulations succeeded
            if sim_success_count > 0:
                pass_fast = check_suppression_maintained(y_max, t_vec, threshold, m_end)
                pass_slow = check_suppression_maintained(y_min, t_vec, threshold, m_end)
            
            return pass_best, pass_fast, pass_slow

        # Binary Search for 'Best'
        low, high = 1, MAX_HOURS
        best_h = 0
        while low <= high:
            mid = (low + high) // 2
            p_best, _, _ = evaluate_h_interval(mid)
            if p_best:
                best_h = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Binary Search for 'Fast' (Least Suppressed)
        low, high = 1, MAX_HOURS
        fast_h = 0
        while low <= high:
            mid = (low + high) // 2
            _, p_fast, _ = evaluate_h_interval(mid)
            if p_fast:
                fast_h = mid
                low = mid + 1
            else:
                high = mid - 1

        # Binary Search for 'Slow' (Most Suppressed)
        low, high = 1, MAX_HOURS
        slow_h = 0
        while low <= high:
            mid = (low + high) // 2
            _, _, p_slow = evaluate_h_interval(mid)
            if p_slow:
                slow_h = mid
                low = mid + 1
            else:
                high = mid - 1

        results["Best"].append(round(best_h / hours_per_week, 3))
        results["Fast"].append(round(fast_h / hours_per_week, 3))
        results["Slow"].append(round(slow_h / hours_per_week, 3))
        print(f"Result for {dose_mg}mg: Best={results['Best'][-1]}w, Fast={results['Fast'][-1]}w, Slow={results['Slow'][-1]}w")

    # Save unique file for this density
    result_filename = f"../../../Data/SLE_skin_SC_dose_response_{density_label}.json"
    with open(result_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"FINISHED: Results saved to {result_filename}")

def plot_SC_dose_response(SC_dose_response_data):
    """
    Plots gathered SC data: Dose (X) vs. Max Allowed Interval (Y).
    Assumes all_sc_results is a dictionary containing data for 
    densities 1, 10, 80, and 400.
    """
    save_dir = '../../../Results/SLE/Dose_response'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    # Using the Red colormap for consistency with PD themes
    colors = plt.cm.Purples(np.linspace(0.6, 0.9, len(SC_dose_response_data)))
    # Standard 4 linestyles plus a 5th custom style for density 400 if needed
    linestyles = ['--', '-', ':', '-.', (0, (3, 5, 1, 5))]

    for i, (label, dataset) in enumerate(SC_dose_response_data.items()):
        color = colors[i]
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot the uncertainty range (Fast vs Slow responders)
        plt.fill_between(dataset['Dose'], dataset['Fast'], dataset['Slow'], 
                         color=color, alpha=0.3)
        
        # Plot the Best parameter line
        plt.plot(dataset['Dose'], dataset['Best'], linestyle=linestyle, color=color, 
                 linewidth=2, marker='o', label=f"{label} pDCs/mm²")

    plt.xlabel('SC Dose Size [mg]', fontsize=18)
    plt.ylabel('Maximum Interval Between Doses [Weeks]', fontsize=18)
    plt.title('Continuous Suppression of >90% BDCA2 on pDCs in Skin', fontsize=18)
    plt.suptitle("Required Frequency of SC Doses to Sustain Response", fontsize=22, fontweight='bold', x=0.52)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper left', fontsize=16, title='pDC Skin Density', title_fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Cap Y-axis to your max search range (e.g., 60 weeks)
    plt.ylim(0, 16) 
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "SC_dose_response.png"), dpi=600)
    plt.savefig(os.path.join(save_dir, "SC_dose_response.svg"), format='svg')
    plt.close()

def plot_SC_dose_response_inverse(SC_dose_response_data):
    """
    Plots gathered SC data with Dose on X and Frequency (1/Interval) on Y.
    Higher Y values = Higher frequency (more doses per week).
    """
    save_dir = '../../../Results/SLE/Dose_response'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    colors = plt.cm.Purples(np.linspace(0.6, 0.9, len(SC_dose_response_data)))
    linestyles = ['--', '-', ':', '-.']

    for i, (label, dataset) in enumerate(SC_dose_response_data.items()):
        color = colors[i]
        linestyle = linestyles[i % len(linestyles)]
        
        # Convert intervals (weeks) to frequencies (1/weeks)
        # Using numpy to handle element-wise division safely
        best_inv = 1.0 / np.array(dataset['Best'])
        fast_inv = 1.0 / np.array(dataset['Fast'])
        slow_inv = 1.0 / np.array(dataset['Slow'])
        
        # Replace infinity (from division by zero) with 0 for plotting
        best_inv[np.isinf(best_inv)] = 0
        fast_inv[np.isinf(fast_inv)] = 0
        slow_inv[np.isinf(slow_inv)] = 0

        # Plot uncertainty: slow_inv is the lower freq, fast_inv is the higher freq
        plt.fill_between(dataset['Dose'], slow_inv, fast_inv, 
                         color=color, alpha=0.3)
        
        # Plot the Best frequency
        plt.plot(dataset['Dose'], best_inv, linestyle=linestyle, color=color, 
                 linewidth=2, marker='o', label=f"{label} pDCs/mm²")

    plt.xlabel('SC Dose Size [mg]', fontsize=18)
    plt.ylabel('Minimum Dosing-Frequency [Doses/Week]', fontsize=18)
    plt.title('Continuous Suppression of >90% BDCA2 on pDCs in Skin', fontsize=18)
    plt.suptitle("Required Frequency of SC Doses to Sustain Response", fontsize=22, fontweight='bold', x=0.52)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(loc='upper right', fontsize=16, title='pDC Skin Density', title_fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Optional: Use log scale if frequencies span multiple orders of magnitude
    plt.yscale('log') 
    
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, "SC_dose_response_inverse.png"), dpi=600)
    plt.savefig(os.path.join(save_dir, "SC_dose_response_inverse.svg"), format='svg')
    plt.close()

SLE_models = {'1': model_1, '10': model_10, '80': model_80, '400': model_400}
SLE_best_param_sets = {'1': params_1, '10': params_10, '80': params_80, '400': params_400}
SLE_acceptable_param_sets = {'1': acceptable_params_1, '10': acceptable_params_10, '80': acceptable_params_80, '400': acceptable_params_400}

ratio_models = { '10': model_10, '80': model_80, 'HV_high': model_HV_high, 'HV': model_HV}
ratio_best_param_sets = { '10': params_10, '80': params_80, 'HV_high': params_80,'HV': params_80}
ratio_acceptable_param_sets = {'10': acceptable_params_10, '80': acceptable_params_80, 'HV_high': acceptable_params_80, 'HV': acceptable_params_80}

target_density = '400'

# plot_skin_PK_PD_simulations_together(SLE_best_param_sets, SLE_acceptable_param_sets, SLE_models, time_vector_IV)

# plot_skin_PK_simulations(SLE_best_param_sets, SLE_acceptable_param_sets, SLE_models, time_vectors_IV_SC)

# plot_skin_PD_simulations(SLE_best_param_sets, SLE_acceptable_param_sets, SLE_models, time_vectors_IV_SC)

# plot_skin_plasma_AUC_ratio(SLE_best_param_sets, SLE_acceptable_param_sets, SLE_models, time_vector_AUC)

plot_IV_dose_response(IV_dose_response_data)

# plot_skin_plasma_concentration_ratio(ratio_best_param_sets, ratio_acceptable_param_sets, ratio_models, time_vector_ratio)

# simulate_SC_dose_response_frequency(target_density, SLE_best_param_sets, SLE_acceptable_param_sets, SLE_models)

plot_SC_dose_response(SC_dose_response_data)

plot_SC_dose_response_inverse(SC_dose_response_data)

# plot_PD_SC_frequency(SLE_best_param_sets, SLE_acceptable_param_sets, SLE_models, SC_dose_response_data)
