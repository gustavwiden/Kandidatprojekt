import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sund

# Load models
sund.install_model('../../../Models/mPBPK_model.txt')
sund.install_model('../../../Models/mPBPK_SLE_model.txt')
model_HV = sund.load_model("mPBPK_model")
model_SLE = sund.load_model("mPBPK_SLE_model")

# Load data
with open("../../../Data/PK_data.json", "r") as f:
    PK_data = json.load(f)
with open("../../../Data/PD_data.json", "r") as f:
    PD_data = json.load(f)

params_HV = [0.6795956201339274, 0.011536420343864593, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6920233945323367, 0.95, 0.7995175786295078, 0.2, 0.007349224278973848, 2.23, 1.04317488678716, 14000.0]  # HV
params_SLE = [0.6795956201339274, 0.011536420343864593, 2.6, 1.81, 6.299999999999999, 4.37, 2.6, 0.010300000000000002, 0.029600000000000005, 0.08100000000000002, 0.6920233945323367, 0.95, 0.7995175786295078, 0.2, 0.008532364216792725, 1.53, 28.299999999999997, 0.10431748867871599, 14000.0]  # SLE

save_dir = "./PK_PD_HV_SLE_figures"
os.makedirs(save_dir, exist_ok=True)

# --- Define simulation objects as in your code ---
bodyweight = 70  # or whatever value is appropriate

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

model_sims_HV = {
    'IVdose_005_HV': sund.Simulation(models = model_HV, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model_HV, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model_HV, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model_HV, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model_HV, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model_HV, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model_HV, activities = SC_50_HV, time_unit = 'h')
}

model_sims_SLE = {
    'IVdose_005_HV': sund.Simulation(models = model_SLE, activities = IV_005_HV, time_unit = 'h'),
    'IVdose_03_HV': sund.Simulation(models = model_SLE, activities = IV_03_HV, time_unit = 'h'),
    'IVdose_1_HV': sund.Simulation(models = model_SLE, activities = IV_1_HV, time_unit = 'h'),
    'IVdose_3_HV': sund.Simulation(models = model_SLE, activities = IV_3_HV, time_unit = 'h'),
    'IVdose_10_HV': sund.Simulation(models = model_SLE, activities = IV_10_HV, time_unit = 'h'),
    'IVdose_20_HV': sund.Simulation(models = model_SLE, activities = IV_20_HV, time_unit = 'h'),
    'SCdose_50_HV': sund.Simulation(models = model_SLE, activities = SC_50_HV, time_unit = 'h')
}

# --- Plotting loop ---
for dose in PK_data:
    # PK plot
    sim_HV = model_sims_HV[dose]
    sim_SLE = model_sims_SLE[dose]
    timepoints = np.arange(-10, PK_data[dose]["time"][-1] + 0.01, 1)

    sim_HV.simulate(time_vector=timepoints, parameter_values=params_HV, reset=True)
    idx = sim_HV.feature_names.index("PK_sim")
    y_HV = sim_HV.feature_data[:, idx]

    sim_SLE.simulate(time_vector=timepoints, parameter_values=params_SLE, reset=True)
    y_SLE = sim_SLE.feature_data[:, idx]

    plt.figure(figsize=(8, 6))
    plt.plot(timepoints, y_HV, 'b-', label='HV PK_sim (best)')
    plt.plot(timepoints, y_SLE, 'r-', label='SLE PK_sim (best)')
    plt.errorbar(
        PK_data[dose]['time'],
        PK_data[dose]['BIIB059_mean'],
        yerr=PK_data[dose]['SEM'],
        fmt='ko',
        markersize=6,
        capsize=3,
        label='PK Data'
    )
    plt.xlabel('Time [Hours]')
    plt.ylabel('BIIB059 Plasma Concentration (µg/ml)')
    plt.title(f'PK: {dose}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"PK_HV_SLE_{dose}.png"))
    plt.close()

    # PD plot (only if dose in PD_data)
    if dose in PD_data:
        sim_HV_PD = model_sims_HV[dose]
        sim_SLE_PD = model_sims_SLE[dose]
        timepoints_PD = np.arange(-10, PD_data[dose]["time"][-1] + 0.01, 1)

        sim_HV_PD.simulate(time_vector=timepoints_PD, parameter_values=params_HV, reset=True)
        idx_PD = sim_HV_PD.feature_names.index("PD_sim")
        y_HV_PD = sim_HV_PD.feature_data[:, idx_PD]

        sim_SLE_PD.simulate(time_vector=timepoints_PD, parameter_values=params_SLE, reset=True)
        y_SLE_PD = sim_SLE_PD.feature_data[:, idx_PD]

        plt.figure(figsize=(8, 6))
        plt.plot(timepoints_PD, y_HV_PD, 'b-', label='HV PD_sim (best)')
        plt.plot(timepoints_PD, y_SLE_PD, 'r-', label='SLE PD_sim (best)')
        plt.errorbar(
            PD_data[dose]['time'],
            PD_data[dose]['BDCA2_median'],
            yerr=PD_data[dose]['SEM'],
            fmt='ko',
            markersize=6,
            capsize=3,
            label='PD Data'
        )
        plt.xlabel('Time [Hours]')
        plt.ylabel('BDCA2 expression (percentage change from baseline)')
        plt.title(f'PD: {dose}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"PD_HV_SLE_{dose}.png"))
        plt.close()

