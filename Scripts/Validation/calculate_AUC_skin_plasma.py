# Import necessary libraries
import os
import re
import numpy as np
import sund
import json
import matplotlib.pyplot as plt

# Open the mPBPK_model.txt file and read its contents
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# Load PK data phase 1
with open("../../Data/SLE_PK_data_plotting.json", "r") as f:
    PK_data_phase_1 = json.load(f)

# Load PK data phase 2A
with open("../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2A = json.load(f)

# Load PK data phase 2B
with open("../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2B = json.load(f)

# Load final parameters for SLE
with open("../../Models/final_parameters_SLE.json", "r") as f:
    params = json.load(f)

# Install the model
sund.install_model('../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model")

# Assumed bodyweight for a fictional SLE patient
bodyweight = 70

# Create activity objects for each dose
IV_005_HV = sund.Activity(time_unit='h')
IV_005_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_005_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_005_HV']['input']['IV_in']['f']))

IV_03_HV = sund.Activity(time_unit='h')
IV_03_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_03_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_03_HV']['input']['IV_in']['f']))

IV_1_HV = sund.Activity(time_unit='h')
IV_1_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_1_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_1_HV']['input']['IV_in']['f']))

IV_3_HV = sund.Activity(time_unit='h')
IV_3_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_3_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_3_HV']['input']['IV_in']['f']))

IV_10_HV = sund.Activity(time_unit='h')
IV_10_HV.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_10_HV']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_10_HV']['input']['IV_in']['f']))

IV_20_SLE = sund.Activity(time_unit='h')
IV_20_SLE.add_output(sund.PIECEWISE_CONSTANT, "IV_in",  t = PK_data_phase_1['IVdose_20_SLE']['input']['IV_in']['t'],  f = bodyweight * np.array(PK_data_phase_1['IVdose_20_SLE']['input']['IV_in']['f']))

SC_50_HV = sund.Activity(time_unit='h')
SC_50_HV.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_1['SCdose_50_HV']['input']['SC_in']['t'],  f = PK_data_phase_1['SCdose_50_HV']['input']['SC_in']['f'])

SC_50_SLE = sund.Activity(time_unit='h')
SC_50_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2A['SCdose_50_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_50_SLE']['input']['SC_in']['f'])

SC_150_SLE = sund.Activity(time_unit='h')
SC_150_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2A['SCdose_150_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_150_SLE']['input']['SC_in']['f'])

SC_450_SLE = sund.Activity(time_unit='h')
SC_450_SLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2A['SCdose_450_SLE']['input']['SC_in']['t'],  f = PK_data_phase_2A['SCdose_450_SLE']['input']['SC_in']['f'])

SC_50_CLE = sund.Activity(time_unit='h')
SC_50_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2B['SCdose_50_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_50_CLE']['input']['SC_in']['f'])

SC_150_CLE = sund.Activity(time_unit='h')
SC_150_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2B['SCdose_150_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_150_CLE']['input']['SC_in']['f'])

SC_450_CLE = sund.Activity(time_unit='h')
SC_450_CLE.add_output(sund.PIECEWISE_CONSTANT, "SC_in",  t = PK_data_phase_2B['SCdose_450_CLE']['input']['SC_in']['t'],  f = PK_data_phase_2B['SCdose_450_CLE']['input']['SC_in']['f'])

# Create simulation objects for each dose, divided by phase
model_sims_phase_1 = {
    'IVdose_005_HV': sund.Simulation(models = model, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_SLE': sund.Simulation(models = model, activities = IV_20_SLE, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model, activities = SC_50_HV, time_unit = 'h'),
}

model_sims_phase_2A = {
    'SCdose_50_SLE': sund.Simulation(models=model, activities=SC_50_SLE, time_unit='h'),
    'SCdose_150_SLE': sund.Simulation(models=model, activities=SC_150_SLE, time_unit='h'),
    'SCdose_450_SLE': sund.Simulation(models=model, activities=SC_450_SLE, time_unit='h')
}

model_sims_phase_2B = {
    'SCdose_50_CLE': sund.Simulation(models=model, activities=SC_50_CLE, time_unit='h'),
    'SCdose_150_CLE': sund.Simulation(models=model, activities=SC_150_CLE, time_unit='h'),
    'SCdose_450_CLE': sund.Simulation(models=model, activities=SC_450_CLE, time_unit='h'),
}

# Create time vectors for each dose, divided by phase
time_vectors_phase_1 = {exp: np.arange(-10, PK_data_phase_1[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_1}
time_vectors_phase_2A = {exp: np.arange(-10, PK_data_phase_2A[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2A}
time_vectors_phase_2B = {exp: np.arange(-10, PK_data_phase_2B[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2B}

# Define a function to calculate AUC using the trapezoidal rule
def calculate_auc(time, y):
    return np.trapezoid(y, time)

# Define a function to simulate and write AUC ratios to a file
def simulate_and_write_auc_ratios(sim_dicts, time_vectors_dicts, params,
                                  save_dir='../../Results/Validation',
                                  filename='AUC_ratios.txt'):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    results = []  # collect rows for plotting

    with open(file_path, 'w') as f:
        f.write(f"{'Phase':<10}{'Dose':<20}{'Plasma_AUC':>15}{'Skin_AUC':>15}{'AUC_Ratio (%)':>15}\n")

        for phase, model_sims in sim_dicts.items():
            time_vectors = time_vectors_dicts[phase]

            for dose_key, sim in model_sims.items():
                time = time_vectors[dose_key]

                # Simulate
                sim.simulate(time_vector=time, parameter_values=params, reset=True)
                plasma = sim.feature_data[:, 0]
                skin   = sim.feature_data[:, 2]

                # AUCs
                auc_plasma = calculate_auc(time, plasma)
                auc_skin   = calculate_auc(time, skin)

                ratio_pct = 100 * auc_skin / auc_plasma if auc_plasma != 0 else np.nan

                f.write(f"{phase:<10}{dose_key:<20}{auc_plasma:15.4f}{auc_skin:15.4f}{ratio_pct:15.4f}\n")
                print(f"{phase} | {dose_key}: Plasma AUC={auc_plasma:.2f}, Skin AUC={auc_skin:.2f}, Ratio={ratio_pct:.3f}")

                # Save for plotting
                # Label example: "phase_1 | IVdose_005_HV"
                results.append({
                    'phase': phase,
                    'dose_key': dose_key,
                    'ratio_pct': ratio_pct
                })

    return results


# Create dictionaries for simulations and time vectors
sim_dicts = {'phase_1': model_sims_phase_1, 'phase_2A': model_sims_phase_2A, 'phase_2B': model_sims_phase_2B}
time_vectors_dicts = {'phase_1': time_vectors_phase_1, 'phase_2A': time_vectors_phase_2A, 'phase_2B': time_vectors_phase_2B}

# Simulate and write AUC ratios to file
simulate_and_write_auc_ratios(sim_dicts = sim_dicts, time_vectors_dicts = time_vectors_dicts, params = params)

# Run sims + write file, and get results for plotting
results = simulate_and_write_auc_ratios(sim_dicts=sim_dicts,
                                        time_vectors_dicts=time_vectors_dicts,
                                        params=params)

# ---- Bar chart of AUC_skin / AUC_plasma (%) by dose ----
# Use dose_key on x-axis. If you want grouping or sorting, you can refine labels below.
labels = [f"{r['dose_key']}" for r in results]
values = [r['ratio_pct'] for r in results]

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, values)

plt.ylabel('AUC_skin / AUC_plasma (%)')
plt.xlabel('Dose (key)')
plt.title('Skin/Plasma AUC Ratio by Dose')

# Rotate labels for readability
plt.xticks(rotation=45, ha='right')

# Add values on top of bars
for b, v in zip(bars, values):
    if np.isfinite(v):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.1f}",
                 ha='center', va='bottom', fontsize=9)

# Save next to your text output
plot_path = os.path.join('../../Results/Validation', 'AUC_ratio_barplot.png')
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved bar chart to: {plot_path}")
