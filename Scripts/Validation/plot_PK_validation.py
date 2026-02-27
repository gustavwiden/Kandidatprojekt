# Importing the necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import sund
import json

# Open the model file and read its contents
model_path = "../../Models/mPBPK_SLE_model_80_pdc_mm2.txt"
with open(model_path, "r") as f:
    lines = f.readlines()

# Load Validation PK data for both CLE and SLE
with open("../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data_CLE = json.load(f)

with open("../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data_SLE = json.load(f)

# Combine datasets
PK_data = {**PK_data_CLE, **PK_data_SLE}

# Load acceptable and final parameters for SLE
acceptable_params = np.loadtxt("../../Results/Acceptable params/acceptable_params_PL_80_pdc_mm2.csv", delimiter=",").tolist()
with open("../../Models/final_parameters_SLE.json", "r") as f:
    best_params = json.load(f)

# Install and load the model
sund.install_model(model_path)
model = sund.load_model("mPBPK_SLE_model_80_pdc_mm2")

# Define all activity objects
activities = {}
for exp, data in PK_data.items():
    act = sund.Activity(time_unit='h')
    act.add_output('piecewise_constant', "SC_in", 
                   t=data['input']['SC_in']['t'], 
                   f=data['input']['SC_in']['f'])
    activities[exp] = act

# Create simulation objects for all doses
model_sims = {exp: sund.Simulation(models=model, activities=activities[exp], time_unit='h') 
              for exp in PK_data}

# Create time vectors based on the maximum time in the PK_data for each experiment
time_vectors = {exp: np.arange(-10, PK_data[exp]["time"][-1] + 400, 1) for exp in PK_data}

def plot_model_uncertainty_with_validation_data(best_params, acceptable_params, sims, PK_data, time_vectors, save_dir='../../Results/Validation/PK Validation'):
    os.makedirs(save_dir, exist_ok=True)

    # Map experiments to specific display labels
    dose_map = {
        'SCdose_50_CLE': '50 mg SC (CLE)', 'SCdose_150_CLE': '150 mg SC (CLE)', 'SCdose_450_CLE': '450 mg SC (CLE)',
        'SCdose_50_SLE': '50 mg SC (SLE)', 'SCdose_150_SLE': '150 mg SC (SLE)', 'SCdose_450_SLE': '450 mg SC (SLE)'
    }
    
    colors = ['#6d65bf', '#6c5ce7', '#8c7ae6', '#6d65bf', '#6c5ce7', '#8c7ae6']

    for i, experiment in enumerate(PK_data.keys()):
        plt.figure(figsize=(10, 8))
        color = colors[i % len(colors)]
        dose_label = dose_map.get(experiment, experiment)
        
        timepoints_hours = time_vectors[experiment]
        timepoints = timepoints_hours / 168.0  # convert hours to weeks
        y_min = np.full_like(timepoints_hours, 10000)
        y_max = np.full_like(timepoints_hours, -10000)

        # Calculate uncertainty range
        for params in acceptable_params:
            SLE_params = np.delete(params.copy(), [10,15])
            try:
                sims[experiment].simulate(time_vector=timepoints_hours, parameter_values=SLE_params, reset=True)
                y_sim = sims[experiment].feature_data[:, 0]
                y_min = np.minimum(y_min, y_sim)
                y_max = np.maximum(y_max, y_sim)
            except RuntimeError as e:
                if "CV_ERR_FAILURE" in str(e):
                    continue
                else:
                    raise e

        # Plot uncertainty range
        plt.fill_between(timepoints, y_min, y_max, color=color, alpha=0.3, label='Uncertainty')

        # Plot best parameter set
        sims[experiment].simulate(time_vector=timepoints_hours, parameter_values=best_params, reset=True)
        plt.plot(timepoints, sims[experiment].feature_data[:, 0], color=color, linewidth=3, label='Simulation')

        # Plot experimental data
        plt.errorbar(
            np.array(PK_data[experiment]['time']) / 168.0,
            PK_data[experiment]['BIIB059_mean'],
            yerr=PK_data[experiment]['SEM'],
            fmt='X', markersize=8, elinewidth=2, capsize=4, color=color, linestyle='None',
            label='Validation Data'
        )

        # Labels and formatting
        plt.xlabel('Time [weeks]', fontsize=18)
        plt.ylabel('Free Litifilimab Plasma Concentration [µg/ml]', fontsize=18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Adjust title based on whether it is CLE or SLE
        sub_title = 'CLE Patients' if 'CLE' in experiment else 'SLE Patients'
        plt.title(f'PK Simulation in Plasma of {sub_title}', fontsize=18)
        plt.suptitle('Validation of Model Against Phase 2 PK Data', fontsize=22, fontweight='bold', x=0.54)
        
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.legend(title=f'Multiple {dose_label.split(" (")[0]} Doses', title_fontsize=18, fontsize=16, loc='upper right')
        plt.tight_layout()

        # Save the plots
        for fmt in ['svg', 'png']:
            plt.savefig(os.path.join(save_dir, f"PK_validation_with_{experiment}.{fmt}"), format=fmt, dpi=600)
        plt.close()

# Run combined plotting
plot_model_uncertainty_with_validation_data(best_params, acceptable_params, model_sims, PK_data, time_vectors)