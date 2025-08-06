import os
import re
import numpy as np
import sund
import json

# Open the mPBPK_model.txt file and read its contents
with open("../../Models/mPBPK_SLE_model.txt", "r") as f:
    lines = f.readlines()

# # Open the data file and read its contents
with open("../../Data/SLE_PK_data_plotting.json", "r") as f:
    PK_data_phase_1 = json.load(f)

with open("../../Data/SLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2A = json.load(f)

with open("../../Data/CLE_Validation_PK_data.json", "r") as f:
    PK_data_phase_2B = json.load(f)

with open("../../Results/Acceptable params/acceptable_params_SLE.json", "r") as f:
    acceptable_params = json.load(f)

# Install the model
sund.install_model('../../Models/mPBPK_SLE_model.txt')
print(sund.installed_models())

# Load the model object
model = sund.load_model("mPBPK_SLE_model")

# Creating activities for the different doses
bodyweight = 69 # Bodyweight for subject in kg

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


# Create simulation objects for each dose
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

time_vectors_phase_1 = {exp: np.arange(-10, PK_data_phase_1[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_1}
time_vectors_phase_2A = {exp: np.arange(-10, PK_data_phase_2A[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2A}
time_vectors_phase_2B = {exp: np.arange(-10, PK_data_phase_2B[exp]["time"][-1] + 0.01, 1) for exp in PK_data_phase_2B}

SLE_params = [0.5982467918487137, 0.013501146489749132, 2.6, 1.125, 6.986999999999999, 4.368, 2.6, 0.006499999999999998, 0.033800000000000004, 0.08100000000000002, 0.75, 0.95, 0.7467544604963505, 0.2, 0.007326666281272475, 0.9621937056820449, 0.10000000000000002, 5.539999999999999, 5.539999999999999, 2623.9999999999995]

def calculate_auc(time, y):
    return np.trapezoid(y, time)

def simulate_and_write_auc_ratios(sim_dicts, time_vectors_dicts, params, save_dir='../../Results/Validation', filename='AUC_ratios.txt'):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w') as f:
        # Header with fixed widths
        f.write(f"{'Phase':<10}{'Dose':<20}{'Plasma_AUC':>15}{'Skin_AUC':>15}{'AUC_Ratio (%)':>15}\n")
        for phase, model_sims in sim_dicts.items():
            time_vectors = time_vectors_dicts[phase]
            for dose, sim in model_sims.items():
                time = time_vectors[dose]
                sim.simulate(time_vector=time, parameter_values=params, reset=True)
                plasma = sim.feature_data[:, 0]
                skin = sim.feature_data[:, 2]
                auc_plasma = calculate_auc(time, plasma)
                auc_skin = calculate_auc(time, skin)
                ratio = 100 * auc_skin / auc_plasma if auc_plasma != 0 else np.nan
                # Data row with fixed widths
                f.write(f"{phase:<10}{dose:<20}{auc_plasma:15.4f}{auc_skin:15.4f}{ratio:15.4f}\n")
                print(f"{phase} | {dose}: Plasma AUC={auc_plasma:.2f}, Skin AUC={auc_skin:.2f}, Ratio={ratio:.3f}")

sim_dicts = {'phase_1': model_sims_phase_1, 'phase_2A': model_sims_phase_2A, 'phase_2B': model_sims_phase_2B}
time_vectors_dicts = {'phase_1': time_vectors_phase_1, 'phase_2A': time_vectors_phase_2A, 'phase_2B': time_vectors_phase_2B}

simulate_and_write_auc_ratios(sim_dicts = sim_dicts, time_vectors_dicts = time_vectors_dicts, params = SLE_params)
